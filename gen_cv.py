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
        "Formatting:\n"
        "When formatting your cover letter, do the following:\n"
        "- Keep it short, sweet, and to the point.\n"
        "- Include the date along with the job ID.\n"
        "- Use short paragraphs, bolded terms, and bullet points.\n"
        "- Do not indent your paragraphs and instead use a business-style letter format.\n\n"
        "Content:\n"
        "Here are some of the best practices for authoring the content of each section of your cover letter:\n\n"
        "Introduction:\n"
        "Your introduction is your first chance to make an impression. Avoid generic openings that only identify the role and your general interest. Use one of the following strategies:\n"
        "- Tell a story that connects with the company brand. Example: ‘Playing World of Warcraft literally changed my life. Creating my own avatar, completing quests, and interacting with other gamers online helped me grow my friend group and build my creativity. The opportunity to work for Blizzard Entertainment is a dream I’ve had since I was 12.’\n"
        "- Use quantifiable data from your professional accomplishments. Example: ‘I saved $400,000 in unnecessary costs and reduced staff time by 20% as a data analyst for Company X. With my master’s in analytics and my experience in SQL, I believe I can do the same for your company.’\n"
        "- Illustrate company knowledge or interest. Example: ‘With a community of over 381 million users, Spotify is the world’s most popular audio streaming subscription service. To maintain that status and continue to grow, you need someone with product analytics experience and a proven track record of crafting success metrics. I hope to be that person.’\n\n"
        
        "Body:\n"
        "In this section, include the keywords from the job ad description. This is especially important as many applicant tracking systems scan cover letters for relevant keywords. Even if they don’t, you want the reader to quickly assess the value you would add to the company. To make the information easy to read and skim through:\n"
        "- Consider using the t-letter format (job requirements on the left, demonstrating your skills for those requirements on the right).\n"
        "- If not using t-letter format, use short paragraphs with not more than 2-4 sentences with specific examples of your skills from the job posting.\n"
        "- Use bullets, section headings, and bolded words.\n"
        "Example:\n"
        "\"As part of my master’s degree in analytics, I currently lead a 5-member team on an 8-month-long practicum project for Company X. Using real-world data, my team created predictive models to understand customer churn. I would love to use my skills to help drive profitable growth for HP:\n"
        "- Statistical understanding-- Proficient in R, regression analyses, and statistical modeling.\n"
        "- Visualization-- Certified in Tableau, adept at creating visually pleasing dashboards.\n"
        "- Communication-- Accomplished in deriving insight and presenting technical information to business audiences.\"\n\n"
        
        "Closing:\n"
        "Focus on connecting to the brand/company by reiterating your specific interests in their work and why you’re a good fit for the job role.\n"
        "Include a few lines asking for further information or steps, in addition to a forward-looking statement.\n\n"
        "Cover Letter:"
    )

    result = generator(prompt, max_length=5000, num_beams=10, early_stopping=True)
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
