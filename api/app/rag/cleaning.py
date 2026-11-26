import re


# A numbered section whose title starts with "Internal:", up to the next numbered section (or end of page)
INTERNAL_SECTION = re.compile(r"^\d+\.\s*Internal:.*?(?=^\d+\.\s|\Z)", re.MULTILINE | re.DOTALL)


def remove_internal(text: str) -> str:
    """Staff-only sections must never be indexed for the customer agent."""
    return INTERNAL_SECTION.sub("", text)
