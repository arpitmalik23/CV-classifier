import os
import win32com.client

def convert_to_pdf(doc_path, pdf_path):
    try:
        word = win32com.client.Dispatch('Word.Application')
        doc = word.Documents.Open(doc_path)
        doc.SaveAs(pdf_path, FileFormat=17)  # FileFormat 17 represents PDF
        doc.Close()
        word.Quit()
        return True
    except Exception as e:
        print(f"Error converting {doc_path} to PDF: {e}")
        return False

def main():
    directory = r"C:\Users\Nityam\Downloads\piramal\CV test"  # Your directory path here
    for filename in os.listdir(directory):
        if filename.endswith((".doc", ".docx")):  # Convert Word documents to PDF
            doc_path = os.path.join(directory, filename)
            pdf_path = os.path.join(directory, os.path.splitext(filename)[0] + ".pdf")
            if not os.path.exists(pdf_path):
                if convert_to_pdf(doc_path, pdf_path):
                    print(f"{filename} converted to PDF successfully.")
                else:
                    print(f"Failed to convert {filename} to PDF.")

if __name__ == "__main__":
    main()
