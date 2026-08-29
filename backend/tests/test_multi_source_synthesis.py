from __future__ import annotations

from app.ai.multi_source_synthesis import build_multi_source_evidence, synthesize_multi_source_answer


def test_multi_source_synthesis_matches_category_to_supplier_evidence_without_raw_dump() -> None:
    evidence = build_multi_source_evidence(
        "Which product categories had the weakest sales performance in August 2026, and do supplier issues explain them?",
        [
            {
                "tool_name": "analytics_summary",
                "status": "success",
                "output": {
                    "summary_type": "category_performance",
                    "data": {
                        "start_date": "2026-08-01",
                        "end_date": "2026-08-31",
                        "ranking_metric": "August revenue, ranked from lowest to highest among returned categories",
                        "items": [
                            {
                                "category_name": "Household",
                                "revenue_cents": 8800000,
                                "units_sold": 320,
                                "gross_margin_cents": 1800000,
                            },
                            {
                                "category_name": "Beauty",
                                "revenue_cents": 9100000,
                                "units_sold": 280,
                                "gross_margin_cents": 2400000,
                            },
                        ],
                    },
                },
            }
        ],
        [
            {
                "source_id": "supplier_report_aug_2026",
                "chunk_id": "12",
                "source_title": "Supplier Performance August 2026",
                "content": (
                    "RetailData-Pro | Supplier Performance | August 2026 Page 1 "
                    "NorthStar Home & Living had 84.3% fulfillment reliability. "
                    "Affected products included paper towels, trash bags, aluminum foil, and detergent. "
                    "Recommendation: split replenishment across backup suppliers."
                ),
            }
        ],
    )

    answer = synthesize_multi_source_answer(evidence)

    assert evidence.matches[0].database_entity == "Household"
    assert evidence.matches[0].document_chunk_id == "12"
    assert "Using August revenue" in answer
    assert "Household" in answer
    assert "NorthStar Home & Living" in answer
    assert "paper towels" in answer
    assert "Beauty" in answer
    assert "no supplier-report evidence directly linking" in answer
    assert "correlation rather than proving causation" in answer
    assert "caused" not in answer.lower()
    assert "Retail database evidence from" not in answer
    assert "Supplier report evidence:" not in answer
    assert "RetailData-Pro | Supplier Performance | August 2026 Page" not in answer
