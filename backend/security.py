import re

INJECTION_PATTERNS = [
    r"ignore (all |previous |your )?instructions",
    r"you are now",
    r"new instructions",
    r"system prompt",
    r"disregard (all |previous )?",
    r"forget (all |everything|your )?",
    r"act as",
    r"jailbreak",
    r"do anything now",
]

OUTPUT_RED_FLAGS = [
    r"my (new |updated )?instructions",
    r"i (have been|am now) (reprogrammed|updated|changed)",
    r"ignore (my |the )?previous",
    r"system prompt is",
]

def sanitize_content(text: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    return text

def validate_output(text: str) -> tuple[bool, str]:
    for pattern in OUTPUT_RED_FLAGS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "I detected potentially unsafe content and cannot respond to this query."
    return True, text