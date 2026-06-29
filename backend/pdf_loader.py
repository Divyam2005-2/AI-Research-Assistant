import fitz

def extract_text_from_pdf(pdf_file):
    # Get the uploaded PDF as bytes
    pdf_bytes = pdf_file.getvalue()

    # Open the PDF from bytes
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text