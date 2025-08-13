import re

# Hook for BD-specific normalization rules
BD_RULES = [
    # Example: normalize retroflex/dental preferences or common orthographic variants
    (r"ড়", "র"),
    (r"ঢ়", "র"),
    (r"\s+", " "),  # collapse spaces
]

def bd_text_normalize(text: str) -> str:
    t = text.strip()
    for pat, rep in BD_RULES:
        t = re.sub(pat, rep, t)
    return t