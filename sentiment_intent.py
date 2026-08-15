"""
sentiment_intent.py
--------------------
Sentiment and intent analysis of user chat queries.

Sentiment: VADER (Valence Aware Dictionary and sEntiment Reasoner), a
lexicon- and rule-based sentiment model that is fast, needs no training,
and is well suited to short, informal text such as chat queries.

Intent: a lightweight rule/keyword-based classifier that buckets a query
into one of a small set of intents relevant to a document-assistant app
(question, summary_request, search_request, greeting, complaint,
gratitude, other). This is intentionally simple and explainable rather
than a black-box classifier, which keeps it free, fast, and dependency-light.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class SentimentResult:
    label: str          # "positive" | "negative" | "neutral"
    compound: float      # VADER compound score in [-1, 1]
    scores: dict          # full pos/neu/neg/compound breakdown


@dataclass
class IntentResult:
    label: str
    matched_keywords: list


_INTENT_PATTERNS = {
    "greeting": [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|afternoon|evening)\b"],
    "gratitude": [r"\bthanks?\b", r"\bthank you\b", r"\bappreciate it\b"],
    "summary_request": [
        r"\bsummar(y|ise|ize|ise)\b", r"\btl;?dr\b", r"\bkey points\b", r"\bmain points\b",
        r"\bgist\b", r"\bshort(er)? version\b", r"\bin brief\b",
    ],
    "search_request": [
        r"\bfind\b", r"\bsearch\b", r"\blook up\b", r"\blocate\b", r"\bwhere (is|are)\b",
    ],
    "complaint": [
        r"\bnot working\b", r"\bwrong\b", r"\bincorrect\b", r"\bbroken\b", r"\bissue\b",
        r"\bproblem\b", r"\bdoesn'?t work\b", r"\bfrustrat\w*\b", r"\bbad answer\b",
    ],
    "question": [
        r"^\s*(what|why|how|when|where|who|which|is|are|can|does|do|did|could|would|should)\b",
        r"\?\s*$",
    ],
}


def analyze_sentiment(text: str) -> SentimentResult:
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return SentimentResult(label=label, compound=compound, scores=scores)


def analyze_intent(text: str) -> IntentResult:
    text_lower = text.lower().strip()

    # Check in a priority order: greeting/gratitude/complaint are usually
    # unambiguous signals; summary/search are specific; question is the
    # broad fallback for anything phrased as a query.
    for intent in ["greeting", "gratitude", "complaint", "summary_request", "search_request"]:
        matched = [p for p in _INTENT_PATTERNS[intent] if re.search(p, text_lower)]
        if matched:
            return IntentResult(label=intent, matched_keywords=matched)

    matched = [p for p in _INTENT_PATTERNS["question"] if re.search(p, text_lower)]
    if matched:
        return IntentResult(label="question", matched_keywords=matched)

    return IntentResult(label="other", matched_keywords=[])


def analyze_query(text: str) -> dict:
    """Convenience combined call used by the Streamlit app / RAG pipeline."""
    sentiment = analyze_sentiment(text)
    intent = analyze_intent(text)
    return {
        "sentiment": sentiment.label,
        "sentiment_compound": sentiment.compound,
        "sentiment_scores": sentiment.scores,
        "intent": intent.label,
        "intent_matched": intent.matched_keywords,
    }
