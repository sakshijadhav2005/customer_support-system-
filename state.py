from typing import TypedDict


class SupportState(TypedDict):
    user_query: str
    category: str  # faq | order_status | complaint | technical_issue | other
    urgency: str  # low | medium | high
    retrieved_docs: list
    response: str
    route: str  # escalation | retrieval | direct | off_topic
