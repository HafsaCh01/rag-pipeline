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
    if text is None:
        return True
    if isinstance(text, float):
        return True
    if not isinstance(text, str):
        return True
    return text.strip() == ""


def fix_encoding(text: str) -> str:
    if not isinstance(text, str) or text == "":
        return text
    fixed = ftfy.fix_text(text)
    fixed = CONTROL_CHAR_RE.sub("", fixed)
    return fixed


def strip_html(text: str) -> str:
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
    if not isinstance(text, str) or text == "":
        return text
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str) or text == "":
        return text
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str, replacement: str = "") -> str:
    if not isinstance(text, str) or text == "":
        return text
    return URL_RE.sub(replacement, text).strip()


def detect_language_safe(text: str):
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

    lang = detect_language_safe(cleaned)
    result["detected_language"] = lang

    if drop_foreign_sentences and lang is not None:
        primary_text, other_text = split_mixed_language(cleaned, primary_lang=primary_lang)
        result["was_mixed_language"] = bool(other_text.strip())
        final_text = primary_text.strip() if primary_text.strip() else cleaned
    else:
        final_text = cleaned

    result["clean_text"] = normalize_whitespace(final_text)
    return result