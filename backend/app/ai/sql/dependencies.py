from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.ai.dependencies import get_ai_provider
from app.ai.provider import AIProvider
from app.ai.sql.pipeline import TextToSQLPipeline
from app.database.session import get_readonly_db


def get_text_to_sql_pipeline(
    ai_provider: AIProvider = Depends(get_ai_provider),
    readonly_db: Session | None = Depends(get_readonly_db),
) -> TextToSQLPipeline | None:
    if readonly_db is None:
        return None
    return TextToSQLPipeline(ai_provider, db=readonly_db)
