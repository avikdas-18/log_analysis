"""
CLI for querying CAN logs in natural language.

Usage:
    python main.py
"""

from query_engine import QueryEngine


def main():
    qe = QueryEngine(db_path="can_logs.db", rules_path="rules.yaml")
    print("CAN Log AI Query Tool. Type your question, or 'quit' to exit.\n")

    while True:
        user_query = input("Query> ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("quit", "exit"):
            break

        try:
            result = qe.query(user_query)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(f"\n[{result['source']}]")
        print(f"Condition used: {result['spec']}\n")

        if not result["matching_logs"]:
            print("No matching logs found.\n")
            continue

        for log_file, readings in result["matching_logs"].items():
            print(f"  {log_file}  ({len(readings)} matching readings)")
            for r in readings[:5]:
                print(f"      t={r['timestamp']:.2f}  {r['signal']}={r['value']}")
            if len(readings) > 5:
                print(f"      ... and {len(readings) - 5} more")

        summary = qe.summarize(user_query, result)
        print(f"\nSummary: {summary}\n")


if __name__ == "__main__":
    main()
