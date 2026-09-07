"""
Generate business-readable answers from verified database results.

The LLM receives:
1. The user's original question
2. The verified SQL
3. The actual database result

The database result is authoritative.
The LLM must not invent or modify numerical values.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class ResponseGenerator:
    """
    Convert verified database results into concise,
    business-readable responses.
    """

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL")

        if not self.api_key:
            raise ValueError("LLM_API_KEY is missing.")

        if not self.model:
            raise ValueError("LLM_MODEL is missing.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def generate_answer(
        self,
        question: str,
        sql: str,
        result
    ) -> str:
        """
        Generate a business-readable answer using ONLY
        the verified database result.
        """

        prompt = f"""
You are a factual business analytics assistant for an AI-powered
Credit Risk Intelligence Platform.

Answer the user's question using ONLY the verified database result
provided below.

============================================================
STRICT FACTUAL RULES
============================================================

1. The verified database result is authoritative.

2. Do not invent numbers, categories, statistics, or facts.

3. Do not change numerical values from the database result.

4. Do not perform additional calculations unless they are necessary
   to directly answer the user's question and are fully supported
   by the verified result.

5. Do not claim information that is not present in the result.

6. Do not generate SQL.

7. Do not discuss information outside the supplied result.

8. If the result is empty, clearly state that no matching records
   were found.

9. Round percentages to two decimal places when appropriate.

10. Keep the answer concise, clear, and business-readable.

============================================================
RESPONSIBLE CREDIT-RISK INTERPRETATION
============================================================

11. Describe patterns as historical associations observed in the
    dataset, not as causal relationships.

12. Do NOT claim that a demographic, educational, occupational,
    financial, or other characteristic causes default.

13. Do NOT recommend automatically approving or rejecting applicants
    based solely on a group characteristic.

14. Do NOT recommend discriminatory lending policies.

15. Do NOT recommend changing loan pricing, approval criteria,
    credit limits, or lending terms solely because of a group-level
    descriptive statistic.

16. When discussing differences between groups, use wording such as:

    "The dataset shows an observed difference..."

    "The historical default rate was..."

    "This may be useful as a monitoring or analytical signal..."

17. Do not describe a group as inherently "high risk" based only
    on an aggregate historical default rate.

18. Distinguish between:
    - historical default rate
    - model-predicted probability
    - business decision

    These are not the same thing.

============================================================
BUSINESS INTERPRETATION
============================================================

19. Highlight the most important pattern directly supported by the
    verified result.

20. If the question asks for a ranking, present the highest-ranked
    results clearly.

21. If the result contains grouped statistics, a small Markdown
    table may be used.

22. Keep recommendations conservative and analytical.

23. Do not recommend a lending action based solely on descriptive
    group statistics.

24. If appropriate, conclude that the observed pattern should be
    evaluated alongside other relevant applicant-level factors
    and the model's overall prediction.

============================================================
USER QUESTION
============================================================

{question}

============================================================
VERIFIED SQL
============================================================

{sql}

============================================================
VERIFIED DATABASE RESULT
============================================================

{result}

Return only the final business-readable answer.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual business analytics "
                        "answering assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()


if __name__ == "__main__":

    print("=" * 80)
    print("BUSINESS RESPONSE GENERATOR TEST")
    print("=" * 80)

    test_question = "Which occupation types have the highest default rates?"

    test_sql = """
    SELECT
        OCCUPATION_TYPE,
        COUNT(*) AS total_applicants,
        SUM(TARGET) AS defaults,
        CAST(SUM(TARGET) AS DOUBLE) / COUNT(*) * 100 AS default_rate
    FROM applicants
    WHERE OCCUPATION_TYPE IS NOT NULL
    GROUP BY OCCUPATION_TYPE
    ORDER BY default_rate DESC
    """

    test_result = """
    OCCUPATION_TYPE          total_applicants    defaults    default_rate
    Low-skill Laborers       2093                359         17.152413
    Drivers                  18603               2107        11.326130
    Accountants              9813                474         4.830327
    """

    generator = ResponseGenerator()

    answer = generator.generate_answer(
        question=test_question,
        sql=test_sql,
        result=test_result
    )

    print("\nQuestion:")
    print(test_question)

    print("\nBusiness Answer:")
    print("-" * 80)
    print(answer)