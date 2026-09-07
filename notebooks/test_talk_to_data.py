import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.talk_to_data.talk_to_data import TalkToData

QUESTIONS = [
    "What percentage of applicants defaulted?",

    "What is the default rate by education type?",

    "What is the default rate by age group?",

    "What is the average income of applicants who defaulted?",

    "Which occupation types have the highest default rates?"
]


def main():

    pipeline = TalkToData()

    print("=" * 100)
    print("TALK-TO-DATA - MULTI QUERY TEST")
    print("=" * 100)

    for i, question in enumerate(QUESTIONS, start=1):

        print("\n" + "=" * 100)
        print(f"QUERY {i}")
        print("=" * 100)

        print("\nQuestion:")
        print(question)

        try:
            response = pipeline.ask(question)

            print("\nGenerated SQL:")
            print(response["sql"])

            print("\nValidation:")
            print(response["validation"]["message"])

            print("\nDatabase Result:")
            print(response["result"])

            print("\nBusiness Answer:")
            print("-" * 80)
            print(response["answer"])

        except Exception as exc:

            print("\nERROR:")
            print(exc)


if __name__ == "__main__":
    main()