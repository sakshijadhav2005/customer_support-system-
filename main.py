from dotenv import load_dotenv

load_dotenv()  # must run before the graph is imported

from graph import app  # noqa: E402

TEST_QUERIES = [
    "What is your return policy?",
    "How do I reset my password?",
    "Where is my order #A1023?",
    "My order is 3 weeks late and I want a refund!",
    "Do you sell gift cards in Antarctica?",
    # edge cases
    "",
    "hi",
    "How much is shipping and how long does it take?",
]


def main():
    print("Graph diagram (Mermaid):")
    print(app.get_graph().draw_mermaid())
    for query in TEST_QUERIES:
        result = app.invoke(
            {
                "user_query": query,
                "category": "",
                "urgency": "",
                "retrieved_docs": [],
                "response": "",
                "route": "",
            }
        )
        print("=" * 60)
        print(f"Query:    {query!r}")
        print(f"Category: {result['category']}")
        print(f"Urgency:  {result['urgency']}")
        print(f"Route:    {result['route']}")
        print(f"Reply:    {result['response']}")


if __name__ == "__main__":
    main()
