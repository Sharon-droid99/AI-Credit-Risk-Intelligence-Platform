import re
import duckdb
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATABASE_PATH = Path("data/credit_risk.duckdb")


# ============================================================
# ALLOWED SCHEMA
# ============================================================

ALLOWED_TABLES = {
    "applicants"
}


# ============================================================
# FORBIDDEN SQL OPERATIONS
# ============================================================

FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
    "COPY",
    "EXPORT",
    "IMPORT"
}


# ============================================================
# LOAD VALID SCHEMA
# ============================================================

def get_allowed_columns():

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True
    )

    try:

        schema = connection.execute(
            """
            DESCRIBE applicants
            """
        ).fetchdf()

        return set(
            schema["column_name"]
        )

    finally:

        connection.close()


# ============================================================
# BASIC SQL CLEANING
# ============================================================

def clean_sql(sql):

    if not isinstance(sql, str):
        raise ValueError(
            "SQL must be provided as a string."
        )

    sql = sql.strip()

    # Remove markdown SQL code fences
    sql = re.sub(
        r"^```(?:sql)?\s*",
        "",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"\s*```$",
        "",
        sql
    )

    return sql.strip()


# ============================================================
# VALIDATE SQL
# ============================================================

def validate_sql(sql):

    try:

        sql = clean_sql(sql)

        if not sql:
            return False, "SQL query is empty."

        # ----------------------------------------------------
        # Prevent multiple statements
        # ----------------------------------------------------

        statements = [
            statement.strip()
            for statement in sql.split(";")
            if statement.strip()
        ]

        if len(statements) != 1:

            return (
                False,
                "Only one SQL statement is allowed."
            )

        sql = statements[0]

        # ----------------------------------------------------
        # Must begin with SELECT or WITH
        # ----------------------------------------------------

        if not re.match(
            r"^(SELECT|WITH)\b",
            sql,
            flags=re.IGNORECASE
        ):

            return (
                False,
                "Only SELECT queries are allowed."
            )

        # ----------------------------------------------------
        # Forbidden operations
        # ----------------------------------------------------

        upper_sql = sql.upper()

        for keyword in FORBIDDEN_KEYWORDS:

            if re.search(
                rf"\b{keyword}\b",
                upper_sql
            ):

                return (
                    False,
                    f"Forbidden SQL operation: {keyword}"
                )

        # ----------------------------------------------------
        # Restrict table usage
        # ----------------------------------------------------

        tables = re.findall(
            r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
            sql,
            flags=re.IGNORECASE
        )

        for table in tables:

            if table.lower() not in ALLOWED_TABLES:

                return (
                    False,
                    f"Table '{table}' is not allowed."
                )

        # ----------------------------------------------------
        # Validate SQL against DuckDB parser
        # ----------------------------------------------------

        connection = duckdb.connect(
            str(DATABASE_PATH),
            read_only=True
        )

        try:

            connection.execute(
                f"EXPLAIN {sql}"
            )

        except Exception as error:

            return (
                False,
                f"Invalid SQL: {error}"
            )

        finally:

            connection.close()

        return True, "SQL query is valid."

    except Exception as error:

        return False, str(error)


# ============================================================
# TEST QUERIES
# ============================================================

def run_tests():

    print("=" * 80)
    print("SQL VALIDATOR TEST")
    print("=" * 80)

    test_queries = [

        # Valid
        (
            "Valid SELECT",
            """
            SELECT
                COUNT(*) AS applicants
            FROM applicants
            """
        ),

        # Valid aggregation
        (
            "Valid GROUP BY",
            """
            SELECT
                NAME_EDUCATION_TYPE,
                COUNT(*) AS applicants,
                AVG(TARGET) AS default_rate
            FROM applicants
            GROUP BY NAME_EDUCATION_TYPE
            """
        ),

        # Invalid operation
        (
            "Forbidden DELETE",
            """
            DELETE FROM applicants
            """
        ),

        # Invalid table
        (
            "Unknown table",
            """
            SELECT *
            FROM customers
            """
        ),

        # Invalid column
        (
            "Unknown column",
            """
            SELECT
                customer_credit_score
            FROM applicants
            """
        ),

        # Multiple statements
        (
            "Multiple statements",
            """
            SELECT COUNT(*) FROM applicants;
            DROP TABLE applicants;
            """
        )
    ]

    for name, query in test_queries:

        valid, message = validate_sql(query)

        print("\n" + "-" * 80)
        print(name)

        print(
            f"Valid : {valid}"
        )

        print(
            f"Message: {message}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_tests()