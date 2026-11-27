from langchain.agents.middleware import PIIMiddleware


# Pakistani formats: 35202-1234567-1 (dashes optional)
CNIC = r"\b\d{5}-?\d{7}-?\d\b"
# 0300-1234567, 03001234567, +923001234567, 00923001234567
PK_PHONE = r"(?:\+92|0092|\b0)3\d{2}[\s-]?\d{7}\b"


def pii_guardrails():
    """Redact customer PII from messages before they reach the model."""
    pass
