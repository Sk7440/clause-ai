from pydantic import Field

from app.schemas.base import Schema


class Source(Schema):
    """One handbook chunk the retriever returned."""

    source: str = Field(default="unknown", description="File the chunk came from")
    page: int | None = Field(default=None, ge=0, description="Page number in the source file")
    score: float = Field(description="Retriever relevance score (rounded to 4 dp)")
