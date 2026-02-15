from io import BytesIO


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(file_bytes)
    if lower.endswith(".docx"):
        return _extract_docx(file_bytes)
    if lower.endswith(".txt"):
        return _extract_txt(file_bytes)
    raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")


def _extract_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf not installed. Run: pip install -r requirements.txt") from exc

    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No readable text found in PDF.")
    return text


def _extract_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "python-docx not installed. Run: pip install -r requirements.txt"
        ) from exc

    doc = Document(BytesIO(file_bytes))
    text = "\n".join(p.text for p in doc.paragraphs if p.text).strip()
    if not text:
        raise ValueError("No readable text found in DOCX.")
    return text


def _extract_txt(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = file_bytes.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode TXT file.")

