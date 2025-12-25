import fitz  # pymupdf


def read_pdf_text(pdf_path: str) -> str:
    """
    读取 PDF 文本内容
    """
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n".join(texts)
