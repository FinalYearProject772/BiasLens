import pdfplumber
from docx import Document


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_file.seek(0)  # Reset file pointer
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    return text.strip() if text.strip() else "No text extracted from PDF"


def extract_text_from_txt(txt_file):
    try:
        txt_file.seek(0)
        text = txt_file.read().decode("utf-8")
        return text if text.strip() else "No text found in file"
    except Exception as e:
        raise Exception(f"Error reading TXT: {str(e)}")


def extract_text_from_docx(uploaded_file):
    try:
        uploaded_file.seek(0)
        doc = Document(uploaded_file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        result = "\n".join(text)
        return result if result.strip() else "No text extracted from DOCX"
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")