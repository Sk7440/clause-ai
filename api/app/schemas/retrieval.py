from app.schemas.base import Schema


class Source(Schema):
    title: str
    content: str
    score: float | None = None