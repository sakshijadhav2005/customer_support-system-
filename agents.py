import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from retriever import retrieve_entries
from state import SupportState

CATEGORIES = ["faq", "order_status", "complaint", "technical_issue", "other", "off_topic"]
URGENCIES = ["low", "medium", "high"]

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        model = os.getenv("LLM_MODEL")
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            from langchain_google_genai import ChatGoogleGenerativeAI

            _llm = ChatGoogleGenerativeAI(
                model=model or "gemini-3.5-flash-lite",
                temperature=0,
                timeout=20,
                max_retries=1,
                google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            )
        elif os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq

            _llm = ChatGroq(model=model or "llama-3.1-8b-instant", temperature=0)
        else:
            _llm = ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    return _llm


CLASSIFIER_PROMPT = (
    "You classify messages sent to the customer support chat of ShopEasy, an "
    "online store. Choose exactly one category from: "
    f"{', '.join(CATEGORIES)}.\n"
    "- faq: a question about store policies or how something works\n"
    "- order_status: asking where an order is or what its status is\n"
    "- complaint: unhappy about an order, product or service, or asking for a refund\n"
    "- technical_issue: a problem with the website, app or login\n"
    "- other: a greeting, or a store-related message that fits none of the above, "
    "including whether the store sells or offers something\n"
    "- off_topic: clearly NOT about the store or shopping at all, such as general "
    "knowledge, news, sports, maths, writing code, jokes, opinions, or personal chat. "
    "If the message could plausibly relate to the store, do NOT use off_topic.\n"
    f"Choose exactly one urgency from: {', '.join(URGENCIES)}. Reply ONLY with JSON "
    'in this format: {"category": "<category>", "urgency": "<urgency>"}'
)

OFF_TOPIC_REPLY = (
    "I'm the ShopEasy customer support assistant, so I can only help with store "
    "questions: orders and shipping, returns and refunds, payments and gift cards, "
    "your account and login, products, and problems with the website or app. "
    "Please ask me about one of those and I'll be happy to help."
)


def parse_classification(text: str) -> dict:
    """Extract and validate the JSON; fall back to safe defaults."""
    category, urgency = "other", "medium"
    match = re.search(r"\{.*?\}", text or "", re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            cat = str(data.get("category", "")).strip().lower()
            urg = str(data.get("urgency", "")).strip().lower()
            if cat in CATEGORIES:
                category = cat
            if urg in URGENCIES:
                urgency = urg
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"category": category, "urgency": urgency}


def classify(state: SupportState) -> dict:
    try:
        reply = get_llm().invoke(
            [SystemMessage(content=CLASSIFIER_PROMPT),
             HumanMessage(content=state["user_query"])]
        ).content
    except Exception:
        reply = ""
    return parse_classification(reply)


def retrieve_faq(state: SupportState) -> dict:
    return {"retrieved_docs": retrieve_entries(state["user_query"]), "route": "retrieval"}


def generate_response(state: SupportState) -> dict:
    docs = state.get("retrieved_docs") or []
    if docs:
        instruction = (
            "You are a polite customer support assistant. Answer the customer "
            "briefly, using ONLY the FAQ text provided. Do not invent policies."
        )
        context = "\n\n".join(docs)
    else:
        instruction = (
            "You are a polite customer support assistant for the ShopEasy online "
            "store. You only help with store topics: orders, shipping, returns and "
            "refunds, payments, accounts and login, products, and technical "
            "problems with the website or app. No FAQ entry matched. "
            "Any question about whether the store sells, offers, delivers or "
            "supports something (for example gift cards, a product, a country) IS "
            "a store topic. "
            "If the customer's message is unrelated to those topics (general "
            "knowledge, news, sports, coding, opinions, small talk, etc.), do NOT "
            "answer it. Briefly and kindly explain that you are a customer support "
            "assistant, list the topics you can help with, and invite them to ask "
            "about one of them. "
            "If the message is about the store but you have no answer, apologise, "
            "say you could not find it, and offer to connect them with a human "
            "agent. Never invent information. Keep the reply to 2-3 sentences."
        )
        context = "No relevant FAQ entries were found."
    prompt = f"{instruction}\n\nFAQ:\n{context}\n\nCustomer query: {state['user_query']}"
    try:
        reply = get_llm().invoke([HumanMessage(content=prompt)]).content
    except Exception as exc:
        reply = f"Sorry, we hit a technical problem. A human agent will follow up. ({type(exc).__name__})"
    return {"response": reply, "route": state.get("route") or "direct"}


def off_topic(state: SupportState) -> dict:
    """Fixed polite redirect; no model call, so it never answers the off-topic question."""
    return {"route": "off_topic", "response": OFF_TOPIC_REPLY}


def escalate(state: SupportState) -> dict:
    return {
        "route": "escalation",
        "response": (
            "We're sorry for the trouble. Your case has been passed to a human "
            "specialist who will be in touch with you shortly."
        )
    }
