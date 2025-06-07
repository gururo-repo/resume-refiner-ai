import fitz
import docx
import io

def parse_resume_file(file):
    """
    Parse resume file content from various formats (PDF, DOCX, TXT).
    Returns the extracted text content.
    """
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.pdf'):
            try:
                doc = fitz.open(stream=file.read(), filetype='pdf')
                text = ''
                for page in doc:
                    text += page.get_text()
                return text.strip()
            except Exception as e:
                print(f"Error parsing PDF: {str(e)}")
                file.seek(0)
                return file.read().decode('utf-8', errors='ignore')
                
        elif filename.endswith('.docx'):
            try:
                doc = docx.Document(io.BytesIO(file.read()))
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                return text.strip()
            except Exception as e:
                print(f"Error parsing DOCX: {str(e)}")
                file.seek(0)
                return file.read().decode('utf-8', errors='ignore')
                
        else:  # Assume text file
            file.seek(0)
            return file.read().decode('utf-8', errors='ignore').strip()
            
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
        raise ValueError(f"Could not parse file: {filename}") 