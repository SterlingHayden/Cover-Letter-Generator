import argparse
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline

# Hardcoded paths
HARDCODED_RESUME_PATH = "./Resumes/Sterling-Hayden-Resume.pdf"
HARDCODED_OUTPUT_PATH = "./CV-Output/cover_letter.docx"

def load_text_from_pdf(pdf_path):
    """Load text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = " ".join(page.extract_text() for page in reader.pages)
    return text

def create_cover_letter(resume_text, job_description_text, output_path):
    """Generate a cover letter and save it to a file."""
    generator = pipeline("text2text-generation", model="t5-base")
    prompt = (
        f"Write a professional cover letter based on the following resume and job description.\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_description_text}\n\n"
        "Cover Letter:"
    )
    result = generator(prompt, max_length=500, num_beams=5, early_stopping=True)
    cover_letter = result[0]["generated_text"]
    
    # Save to a file
    doc = Document()
    doc.add_paragraph(cover_letter)
    doc.save(output_path)
    print(f"Cover letter saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a cover letter based on a resume and job description.")
    parser.add_argument("job_description", help="Path to the job description file.")
    parser.add_argument("--resume", help="Path to the resume file.", default=HARDCODED_RESUME_PATH)
    parser.add_argument("--output", help="Path to save the generated cover letter.", default=HARDCODED_OUTPUT_PATH)
    
    args = parser.parse_args()

    # Load resume and job description text
    resume_text = load_text_from_pdf(args.resume)
    with open(args.job_description, "r") as job_file:
        job_description_text = job_file.read()

    # Generate and save cover letter
    create_cover_letter(resume_text, job_description_text, args.output)

if __name__ == "__main__":
    main()
