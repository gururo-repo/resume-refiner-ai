import fitz

def parse_resume_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        try:
            doc = fitz.open(stream=file.read(), filetype='pdf')
            text = ''
            for page in doc:
                text += page.get_text()
            return text
        except Exception:
            file.seek(0)
            return file.read().decode('utf-8', errors='ignore')
    else:
        return file.read().decode('utf-8', errors='ignore') 