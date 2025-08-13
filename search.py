import os
import json
import numpy as np
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, max_length=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start = end
    return chunks

def load_chunks_and_embeddings():
    # Load resume PDF text
    resume_text = load_pdf_text("me/Pooja_Nigam_Resume.pdf")
    
    # Load summary text
    with open("me/summary2.txt", "r", encoding="utf-8") as f:
        summary_text = f.read()
    
    combined_text = resume_text + "\n\n" + summary_text
    chunks = chunk_text(combined_text)
    
    # Load embeddings from JSON file
    with open("me/embeddings.json", "r", encoding="utf-8") as f:
        embeddings = json.load(f)
    
    return chunks, embeddings

def find_similar_chunks(question, chunks, embeddings, top_k=3):
    # You should generate embedding for the question using OpenAI embeddings
    # For now, let's assume you pass question_embedding from chatbot
    
    # Placeholder: you need to get question_embedding from your main code
    raise NotImplementedError("You need to pass question embedding to this function from your chatbot")
    
def find_similar_chunks_with_embedding(question_embedding, chunks, embeddings, top_k=3):
    # Convert lists to numpy arrays
    embeddings_np = np.array(embeddings)
    question_emb_np = np.array(question_embedding).reshape(1, -1)
    
    similarities = cosine_similarity(question_emb_np, embeddings_np)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices]

if __name__ == "__main__":
    chunks, embeddings = load_chunks_and_embeddings()
    print(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings.")
