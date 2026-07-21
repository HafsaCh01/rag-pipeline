
import re
import unicodedata
import ftfy
from bs4 import BeautifulSoup
 
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
except ImportError: 
    detect = None
    LangDetectException = Exception
 
 
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
 
 
def is_missing_or_empty(text) -> bool:
    """
    Returns True if text is None, NaN, empty, or whitespace-only.
    Handles pandas NaN (float) safely without importing pandas here.
    """
    if text is None:
        return True
    if isinstance(text, float):  
        return True
    if not isinstance(text, str):
        return True
    return text.strip() == ""
 
 
def fix_encoding(text: str) -> str:
    """
    Repairs mojibake / double-encoded text using ftfy, and strips
    stray control characters. Safe no-op on already-clean text.
    """
    if not isinstance(text, str) or text == "":
        return text
    fixed = ftfy.fix_text(text)
    fixed = CONTROL_CHAR_RE.sub("", fixed)
    return fixed
 
 
def strip_html(text: str) -> str:
    """
    Removes HTML tags, keeping inner text. Falls back to a regex strip
    if BeautifulSoup parsing fails for any reason (e.g. malformed markup).
    """
    if not isinstance(text, str) or text == "":
        return text
    if "<" not in text or ">" not in text:
        return text 
    try:
        extracted = BeautifulSoup(text, "lxml").get_text(separator=" ")
    except Exception:
        extracted = HTML_TAG_RE.sub(" ", text)
  
    return MULTI_SPACE_RE.sub(" ", extracted).strip()
 
 
def normalize_whitespace(text: str) -> str:
    """
    Collapses repeated spaces/tabs into a single space, trims excess
    blank lines, and strips leading/trailing whitespace.
    """
    if not isinstance(text, str) or text == "":
        return text
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
 
 
def normalize_unicode(text: str) -> str:
    """
    Applies NFKC unicode normalization so visually-identical characters
    (e.g. full-width vs half-width, combining accents) are consistent.
    """
    if not isinstance(text, str) or text == "":
        return text
    return unicodedata.normalize("NFKC", text)
 
 
def remove_urls(text: str, replacement: str = "") -> str:
    if not isinstance(text, str) or text == "":
        return text
    return URL_RE.sub(replacement, text).strip()
 
 
def detect_language_safe(text: str):
    """
    Returns an ISO 639-1 language code, or None if detection is
    impossible (too short, ambiguous, or the detector errors out).
    Never raises.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return None
    if detect is None:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None
    except Exception:
        return None
 
 
def split_mixed_language(text: str, primary_lang: str = "en"):
    """
    Best-effort split of a mixed-language document into sentences that
    match the primary language vs. sentences that don't. Uses a naive
    sentence splitter (good enough for a cleaning-stage heuristic; a
    proper pipeline would use a sentence tokenizer per detected script).
 
    Returns (primary_text, other_text). If detection is unavailable or
    the text is short, returns (text, "") unchanged.
    """
    if not isinstance(text, str) or text.strip() == "":
        return text, ""
    if detect is None:
        return text, ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    primary_sents, other_sents = [], []
    for s in sentences:
        if not s.strip():
            continue
        lang = detect_language_safe(s)
        if lang is None or lang == primary_lang:
            primary_sents.append(s)
        else:
            other_sents.append(s)
    return " ".join(primary_sents), " ".join(other_sents)
 
 
def clean_text(text, primary_lang: str = "en", drop_foreign_sentences: bool = True):
    """
    Main entry point: runs the full cleaning pipeline on a single document.
 
    Returns a dict:
        {
            "clean_text": str,          # cleaned text, "" if unrecoverable
            "was_empty": bool,          # True if input was missing/empty
            "had_html": bool,
            "had_encoding_issue": bool,
            "detected_language": str | None,
            "was_mixed_language": bool,
        }
 
    Never raises — always returns a well-formed result dict, which is
    what makes it safe to run over an entire dataset with .apply().
    """
    result = {
        "clean_text": "",
        "was_empty": False,
        "had_html": False,
        "had_encoding_issue": False,
        "detected_language": None,
        "was_mixed_language": False,
    }
 
    if is_missing_or_empty(text):
        result["was_empty"] = True
        return result
 
    original = text
    result["had_html"] = bool(HTML_TAG_RE.search(original))

    fixed = fix_encoding(original)
    result["had_encoding_issue"] = fixed != original
 
    no_html = strip_html(fixed)
    no_urls = remove_urls(no_html)
    normalized = normalize_unicode(no_urls)
    cleaned = normalize_whitespace(normalized)
 
    if cleaned == "":
        result["was_empty"] = True
        result["clean_text"] = ""
        return result
    has_non_ascii = any(ord(c) > 127 for c in cleaned)
 
    lang = detect_language_safe(cleaned)
    result["detected_language"] = lang
 
    if drop_foreign_sentences and lang is not None and has_non_ascii:
        primary_text, other_text = split_mixed_language(cleaned, primary_lang=primary_lang)
        result["was_mixed_language"] = bool(other_text.strip())
        final_text = primary_text.strip() if primary_text.strip() else cleaned
    else:
        final_text = cleaned
 
    result["clean_text"] = normalize_whitespace(final_text)
    return result