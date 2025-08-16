import re

BD_RULES = [
    (r"বাংলাদে\s*শ", "বাংলাদেশ"),
    (r"মানষু", "মানুষ"),
]

PHONEME_MAP = {
    'ড়': 'ɽ',
    'ঢ়': 'ɽʱ',
}

def bd_text_normalize(text: str) -> str:
    t = text
    for pat, rep in BD_RULES:
        t = re.sub(pat, rep, t)
    return t.strip()

def to_phonemes(text: str, lang: str = 'bn') -> str:
    """Try eSpeak via phonemizer; fallback to graphemes on Windows if not available."""
    t = bd_text_normalize(text)
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        if EspeakBackend.is_available():
            ph = phonemize(t, language=lang, backend='espeak', strip=True, njobs=1)
            for k, v in PHONEME_MAP.items():
                ph = ph.replace(k, v)
            return ph
    except Exception:
        pass
    # Fallback: return normalized text (graphemes)
    return t
