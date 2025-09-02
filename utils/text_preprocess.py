import re
import unicodedata

def clean_text(self: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?:])", r"\1", text)
    text = re.sub(r"([!?])\1+", r"\1", text)

    return text