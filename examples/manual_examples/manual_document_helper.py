"""
Minimal helpers for manually adding documents to Co-STORM demos.
"""

from typing import Dict, List, Optional

from knowledge_storm.interface import Information


def create_information_from_document(
    url: str,
    title: str,
    content: str,
    description: Optional[str] = None,
    question: Optional[str] = None,
    query: Optional[str] = None,
    snippets: Optional[List[str]] = None,
) -> Information:
    """
    Create an Information object from free-form content.
    """
    if snippets is None:
        snippets = [s.strip() for s in content.split("\n\n") if s.strip()]
        if not snippets:
            snippets = [f"{s.strip()}." for s in content.split(".") if s.strip()]
        if not snippets:
            snippets = [content.strip()]

    if description is None:
        first = snippets[0]
        description = first[:200] + "..." if len(first) > 200 else first

    meta = {}
    if question:
        meta["question"] = question
    if query:
        meta["query"] = query

    return Information(
        url=url,
        title=title,
        description=description,
        snippets=snippets,
        meta=meta or None,
    )


def create_information_from_dict(doc_dict: Dict) -> Information:
    """
    Create an Information object from a JSON-friendly dictionary.
    """
    return create_information_from_document(
        url=doc_dict["url"],
        title=doc_dict["title"],
        content=doc_dict.get("content", ""),
        description=doc_dict.get("description"),
        question=doc_dict.get("question"),
        query=doc_dict.get("query"),
        snippets=doc_dict.get("snippets"),
    )

