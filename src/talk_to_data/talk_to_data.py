"""
End-to-end Talk-to-Data pipeline.

Flow:
Natural Language Question
        ↓
LLM generates SQL
        ↓
SQL validation
        ↓
DuckDB execution
        ↓
LLM converts verified result into business answer
"""

from src.talk_to_data.nl_to_sql import NLToSQL
from src.talk_to_data.query_runner import execute_query
from src.talk_to_data.sql_validator import validate_sql
from src.talk_to_data.response_generator import ResponseGenerator

class TalkToData:
    def __init__(self):
        self.nl_to_sql = NLToSQL()
        self.response_generator = ResponseGenerator()

    def ask(self, question: str):
        # ---------------------------------------------------------
        # 1. Natural language → SQL
        # ---------------------------------------------------------
        sql = self.nl_to_sql.generate_sql(question)

        # ---------------------------------------------------------
        # 2. Handle unsupported questions
        # ---------------------------------------------------------
        if sql.strip().upper() == "UNSUPPORTED_QUERY":
            return {
                "question": question,
                "sql": sql,
        "validation": {
            "valid": False,
            "message": "The question cannot be answered using the available data."
        },
        "result": None,
        "answer": (
            "I can't answer that question using the currently available "
            "applicant data."
        )
    }

        # ---------------------------------------------------------
        # 2. Validate generated SQL
        # ---------------------------------------------------------
        validation_valid, validation_message = validate_sql(sql)

        validation = {
            "valid": validation_valid,
            "message": validation_message
        }

        if not validation_valid:
            return {
                "question": question,
                "sql": sql,
                "validation": validation,
                "result": None,
                "answer": (
                    "I could not safely process this question because "
                    f"the generated SQL failed validation: {validation_message}"
                )
            }

        # ---------------------------------------------------------
        # 3. Execute verified SQL against DuckDB
        # ---------------------------------------------------------
        try:
            result = execute_query(sql)
        except Exception as exc:
            return {
                "question": question,
                "sql": sql,
                "validation": validation,
                "result": None,
                "answer": (
                    "The query passed validation but could not be "
                    f"executed successfully: {exc}"
                )
            }

        # ---------------------------------------------------------
        # 4. Convert verified result → business-readable answer
        # ---------------------------------------------------------
        answer = self.response_generator.generate_answer(
            question=question,
            sql=sql,
            result=result
        )

        return {
            "question": question,
            "sql": sql,
            "validation": validation,
            "result": result,
            "answer": answer
        }


if __name__ == "__main__":

    print("=" * 80)
    print("CREDIT RISK PLATFORM - END-TO-END TALK TO DATA")
    print("=" * 80)

    question = "What is the average number of mobile phone calls made by applicants?"

    pipeline = TalkToData()

    response = pipeline.ask(question)

    print("\nQuestion:")
    print(response["question"])

    print("\nGenerated SQL:")
    print(response["sql"])

    print("\nValidation:")
    print(response["validation"]["message"])

    print("\nDatabase Result:")
    print(response["result"])

    print("\nBusiness Answer:")
    print("-" * 80)
    print(response["answer"])