import re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF file."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Remove noise: extra whitespace, special chars, blank lines."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # strip non-ASCII
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def load_pdf(path: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    raw = extract_text_from_pdf(path)
    cleaned = clean_text(raw)
    return chunk_text(cleaned, chunk_size, overlap)