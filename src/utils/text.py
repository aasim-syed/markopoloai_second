import re
from phonemizer import phonemize

BD_RULES = [
    # Example BD-specific normalizations and common orthographic variants
    (r"বাংলাদে\s*শ", "বাংলাদেশ"),
    (r"মানষু", "মানুষ"),
]

# Some phoneme mapping adjustments (illustrative)
PHONEME_MAP = {
    'ড়': 'ɽ',  # retroflex flap
    'ঢ়': 'ɽʱ',
}

def bd_text_normalize(text: str) -> str:
    t = text
    for pat, rep in BD_RULES:
        t = re.sub(pat, rep, t)
    return t.strip()


def to_phonemes(text: str, lang: str = 'bn') -> str:
    text = bd_text_normalize(text)
    ph = phonemize(text, language=lang, backend='espeak', strip=True, njobs=1)
    for k, v in PHONEME_MAP.items():
        ph = ph.replace(k, v)
    return ph