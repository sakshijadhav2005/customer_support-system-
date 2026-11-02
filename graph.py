from langgraph.graph import END, START, StateGraph

from agents import classify, escalate, generate_response, off_topic, retrieve_faq
from state import SupportState


def route(state: SupportState) -> str:
    if state["category"] == "off_topic":  # checked first so urgency cannot escalate it
        return "off_topic"
    if state["category"] == "complaint" or state["urgency"] == "high":
        return "escalate"
    if state["category"] in ("faq", "technical_issue"):
        return "retrieve_faq"
    return "generate_response"


def build_graph():
    g = StateGraph(SupportState)
    g.add_node("classify", classify)
    g.add_node("retrieve_faq", retrieve_faq)
    g.add_node("generate_response", generate_response)
    g.add_node("escalate", escalate)
    g.add_node("off_topic", off_topic)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        route,
        {
            "off_topic": "off_topic",
            "escalate": "escalate",
            "retrieve_faq": "retrieve_faq",
            "generate_response": "generate_response",
        },
    )
    g.add_edge("retrieve_faq", "generate_response")
    g.add_edge("generate_response", END)
    g.add_edge("escalate", END)
    g.add_edge("off_topic", END)
    return g.compile()


app = build_graph()
