"""
Natural Language to SQL generation.

This module:
1. Loads the LLM configuration.
2. Retrieves the database schema.
3. Builds the SQL-generation prompt.
4. Sends the question to the LLM.
5. Extracts the generated SQL.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Allow imports from the project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.talk_to_data.prompt_templates import build_sql_prompt
from src.talk_to_data.query_runner import get_schema


load_dotenv()


class NLToSQL:
    """Convert natural-language questions into SQL."""

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL")

        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is missing. "
                "Add it to your local .env file."
            )

        if not self.model:
            raise ValueError(
                "LLM_MODEL is missing. "
                "Add the model name to your local .env file."
            )

        self.client = OpenAI(
        api_key=self.api_key,
        base_url="https://openrouter.ai/api/v1"
        )

        self.schema = get_schema()

    def generate_sql(self, question: str) -> str:
        """
        Convert a natural-language question into SQL.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        system_prompt, user_prompt = build_sql_prompt(
            self.schema,
            question.strip()
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        sql = response.choices[0].message.content.strip()

        # Remove accidental Markdown fences.
        if sql.startswith("```"):
            sql = sql.replace("```sql", "")
            sql = sql.replace("```", "")
            sql = sql.strip()

        return sql


if __name__ == "__main__":

    print("=" * 80)
    print("NATURAL LANGUAGE → SQL TEST")
    print("=" * 80)

    generator = NLToSQL()

    question = "What percentage of applicants defaulted?"

    print(f"\nQuestion: {question}")

    sql = generator.generate_sql(question)

    print("\nGenerated SQL:")
    print("-" * 80)
    print(sql)