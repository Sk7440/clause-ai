from pydantic import BaseModel, Field


class Source(BaseModel):
    "...."

    source: str = Field(default="unknown", description="")
    page: int | None = Field(default=None, description="")
    score: float = Field(description="")
