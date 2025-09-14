import os
from dotenv import load_dotenv
from mistralai import Mistral
from pypdf import PdfReader
import json

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()  # strip whitespace

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def chunk_text(text, max_length=500):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_length, text_length)
        chunk = text[start:end].strip()
        if chunk:  # only add non-empty chunks
            chunks.append(chunk)
        start = end
    return chunks

def create_embeddings(text_chunks):
    embeddings = []
    for i, chunk in enumerate(text_chunks):
        # Add print for debugging chunk length
        print(f"Embedding chunk {i+1}/{len(text_chunks)} (length {len(chunk)})")
        response = client.embeddings.create(
            model="mistral-embed", inputs=[chunk]
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def main():
    resume_text = load_pdf_text("me/Pooja_Nigam_Resume.pdf")
    summary_text = load_text("me/summary2.txt")
    github_text = load_text("me/github_profile.txt")

    combined_text = resume_text + "\n\n" + summary_text + "\n\n" + github_text
    combined_text = combined_text.strip()

    if not combined_text:
        print("Error: Combined text is empty. Check input files.")
        return

    chunks = chunk_text(combined_text)
    print(f"Total chunks created: {len(chunks)}")

    if len(chunks) == 0:
        print("Error: No chunks created from text.")
        return

    embeddings = create_embeddings(chunks)

    if len(chunks) != len(embeddings):
        print(f"Warning: Chunks count ({len(chunks)}) != Embeddings count ({len(embeddings)})")

    with open("me/embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f"Created and saved {len(embeddings)} embeddings.")

if __name__ == "__main__":
    main()
