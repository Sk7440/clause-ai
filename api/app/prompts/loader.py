from pathlib import Path


PROMPTS_DIR = Path(__file__).parent


def get_prompt(name: str, version: str = "v1") -> None:
    return (PROMPTS_DIR / name / f"{version}.md").read_text(encoding="utf-8")


# name : agent_name/prompt type
# name : support /system/v3


# path : api/app/prompts/support/system/v3.md
