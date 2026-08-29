from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class DatabaseEvidence(BaseModel):
    metric_name: str
    entity: str
    period: str
    values: dict[str, Any] = Field(default_factory=dict)
    source: str = "analytics_summary"


class DocumentEvidence(BaseModel):
    source_id: str
    chunk_id: str
    document_name: str
    content_summary: str
    categories: list[str] = Field(default_factory=list)
    supplier: str | None = None
    recommendation: str | None = None


class CrossSourceMatch(BaseModel):
    database_entity: str
    document_source_id: str
    document_chunk_id: str
    relationship: str
    evidence_summary: str


class MultiSourceEvidence(BaseModel):
    question: str
    ranking_metric: str
    database_evidence: list[DatabaseEvidence] = Field(default_factory=list)
    document_evidence: list[DocumentEvidence] = Field(default_factory=list)
    matches: list[CrossSourceMatch] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


CATEGORY_ALIASES = {
    "household": {"household", "paper towel", "paper towels", "trash bag", "trash bags", "aluminum foil", "detergent", "cleaning"},
    "electronics": {"electronics", "charger", "chargers", "headphones", "tablet", "laptop", "device", "devices"},
    "beauty": {"beauty", "cosmetic", "cosmetics", "skin care", "skincare", "shampoo", "conditioner"},
    "grocery": {"grocery", "groceries", "coffee", "snack", "snacks", "cereal", "pantry"},
}


def build_multi_source_evidence(question: str, tool_results: list[dict[str, Any]], retrieved_chunks: list[dict[str, Any]]) -> MultiSourceEvidence:
    database_evidence, ranking_metric = _database_evidence(tool_results)
    document_evidence = [_document_evidence(chunk) for chunk in retrieved_chunks]
    matches = _match_sources(database_evidence, document_evidence)
    limitations: list[str] = []

    if not database_evidence:
        limitations.append("Retail database evidence was unavailable.")
    if not document_evidence:
        limitations.append("Supplier-report evidence was unavailable.")

    return MultiSourceEvidence(
        question=question,
        ranking_metric=ranking_metric,
        database_evidence=database_evidence,
        document_evidence=document_evidence,
        matches=matches,
        limitations=limitations,
    )


def synthesize_multi_source_answer(evidence: MultiSourceEvidence) -> str:
    if not evidence.database_evidence and not evidence.document_evidence:
        return "I could not gather enough verified evidence to answer this question because the required data sources were unavailable."

    parts: list[str] = []
    if evidence.database_evidence:
        entities = evidence.database_evidence[:3]
        entity_names = ", ".join(item.entity for item in entities)
        parts.append(f"Using {evidence.ranking_metric}, the weakest returned categories were {entity_names}.")
        for item in entities:
            match = next((candidate for candidate in evidence.matches if candidate.database_entity == item.entity), None)
            metric_text = _database_metric_text(item)
            if match:
                document = next(
                    doc
                    for doc in evidence.document_evidence
                    if doc.source_id == match.document_source_id and doc.chunk_id == match.document_chunk_id
                )
                supplier_text = f" from {document.supplier}" if document.supplier else ""
                recommendation = f" The report recommendation was: {document.recommendation}." if document.recommendation else ""
                parts.append(
                    f"{item.entity}: {metric_text}. The supplier report has matching evidence{supplier_text}: "
                    f"{match.evidence_summary}. This is consistent with weaker performance, but it shows correlation rather than proving causation."
                    f"{recommendation}"
                )
            else:
                parts.append(
                    f"{item.entity}: {metric_text}. I found no supplier-report evidence directly linking supplier issues to this category's result."
                )
    elif evidence.document_evidence:
        summaries = " ".join(document.content_summary for document in evidence.document_evidence[:2])
        parts.append(f"The supplier report evidence says: {summaries}")

    if evidence.limitations:
        parts.append("Limitations: " + " ".join(evidence.limitations))
    return " ".join(parts)


def _database_evidence(tool_results: list[dict[str, Any]]) -> tuple[list[DatabaseEvidence], str]:
    analytics = next((item for item in tool_results if item.get("tool_name") == "analytics_summary"), None)
    if not analytics:
        return [], "available evidence"

    output = analytics.get("output") or {}
    data = output.get("data") or {}
    summary_type = output.get("summary_type", "analytics")
    metric = data.get("ranking_metric") or _default_metric(summary_type)
    period = _period(data)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return [], metric

    if summary_type == "category_performance":
        items = sorted(items, key=lambda item: int((item or {}).get("revenue_cents") or 0))

    evidence = [
        DatabaseEvidence(
            metric_name=summary_type,
            entity=_entity_name(item),
            period=period,
            values=item,
        )
        for item in items[:3]
        if isinstance(item, dict)
    ]
    return evidence, metric


def _document_evidence(chunk: dict[str, Any]) -> DocumentEvidence:
    content = str(chunk.get("content") or "")
    clean_content = _clean_document_content(content)
    return DocumentEvidence(
        source_id=str(chunk.get("source_id") or ""),
        chunk_id=str(chunk.get("chunk_id") or ""),
        document_name=str(chunk.get("source_title") or chunk.get("title") or "document"),
        content_summary=_relevant_summary(clean_content),
        categories=_categories_for_text(clean_content),
        supplier=_supplier(clean_content),
        recommendation=_recommendation(clean_content),
    )


def _match_sources(database: list[DatabaseEvidence], documents: list[DocumentEvidence]) -> list[CrossSourceMatch]:
    matches: list[CrossSourceMatch] = []
    for item in database:
        aliases = _aliases_for_category(item.entity)
        for document in documents:
            document_categories = {category.lower() for category in document.categories}
            if item.entity.lower() in document_categories or aliases & _normalized_terms(document.content_summary):
                matches.append(
                    CrossSourceMatch(
                        database_entity=item.entity,
                        document_source_id=document.source_id,
                        document_chunk_id=document.chunk_id,
                        relationship="correlated",
                        evidence_summary=document.content_summary,
                    )
                )
                break
    return matches


def _entity_name(item: dict[str, Any]) -> str:
    return str(
        item.get("category_name")
        or item.get("product_name")
        or item.get("supplier_name")
        or item.get("customer_name")
        or item.get("sku")
        or "unknown"
    )


def _default_metric(summary_type: str) -> str:
    if summary_type == "category_performance":
        return "August revenue, ranked from lowest to highest among returned categories"
    return summary_type


def _period(data: dict[str, Any]) -> str:
    start = data.get("start_date")
    end = data.get("end_date")
    if start and end:
        return f"{start} to {end}"
    return "selected period"


def _database_metric_text(item: DatabaseEvidence) -> str:
    values = item.values
    metrics: list[str] = []
    if "revenue_cents" in values:
        metrics.append(f"revenue {_money(values.get('revenue_cents'))}")
    if "units_sold" in values:
        metrics.append(f"{values.get('units_sold')} units sold")
    if "gross_margin_cents" in values:
        metrics.append(f"gross margin {_money(values.get('gross_margin_cents'))}")
    if not metrics:
        metrics.append(f"metric {item.metric_name}")
    return f"{', '.join(metrics)} for {item.period}"


def _clean_document_content(content: str) -> str:
    cleaned = re.sub(r"RetailData-Pro\s*\|\s*Supplier Performance\s*\|\s*August 2026\s*Page\s*\d+", " ", content, flags=re.I)
    cleaned = re.sub(r"RetailData-Pro Supplier Performance Report", " ", cleaned, flags=re.I)
    return " ".join(cleaned.split())


def _relevant_summary(content: str, max_sentences: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", content)
    important = [
        sentence
        for sentence in sentences
        if re.search(r"supplier|reliability|fulfillment|affected|shortage|risk|recommend|lead time|category|product", sentence, re.I)
    ]
    selected = important[:max_sentences] or sentences[:1]
    summary = " ".join(sentence.strip() for sentence in selected if sentence.strip())
    return _truncate(summary, 520)


def _categories_for_text(content: str) -> list[str]:
    normalized = content.lower()
    categories = []
    for category, aliases in CATEGORY_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            categories.append(category.title())
    return categories


def _aliases_for_category(category: str) -> set[str]:
    return CATEGORY_ALIASES.get(category.lower(), {category.lower()})


def _normalized_terms(value: str) -> set[str]:
    normalized = value.lower()
    terms = {term.strip(".,:;!?()[]") for term in normalized.split()}
    for category, aliases in CATEGORY_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            terms.add(category)
            terms.update(aliases)
    return terms


def _supplier(content: str) -> str | None:
    match = re.search(r"([A-Z][A-Za-z& ]{2,80})\s+(?:had|recorded|reported|showed)[^.]{0,120}reliability", content)
    if match:
        return match.group(1).strip()
    return None


def _recommendation(content: str) -> str | None:
    match = re.search(r"(?:recommend(?:ed|s|ation)?|action(?: recommended)?):?\s*([^.\n]+)", content, flags=re.I)
    if match:
        return _truncate(match.group(1).strip(), 220)
    return None


def _money(cents: Any) -> str:
    value = int(cents or 0)
    return f"${value / 100:,.2f}"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
