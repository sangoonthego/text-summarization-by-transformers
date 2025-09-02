import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz

def read_txt(file) -> str:
    file.seek(0)
    try:
        return file.read().decode("utf-8") # convert bytes into Unicode (utf-8) sequence
    except UnicodeDecodeError:
        file.seek(0) # reset pointer
        return file.read().decode("latin-1", errors="ignore")

def read_pdf(file) -> str:
    file.seek(0)
    text = []
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text.append(page.get_text("text"))

    return "\n".join(text)

def read_file(file) -> str:
    name = getattr(file, "name", "")

    if file.type == "text/plain" or name.endswith(".txt"):
        return read_txt(file)
    elif file.type == "application/pdf" or name.endswith(".pdf"):
        return read_pdf(file)
    
    return ""