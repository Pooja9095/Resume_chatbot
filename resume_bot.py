from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from database import get_answer, add_unknown_question, add_qa, get_user, increment_questions, add_user
import sqlite3
ADMIN_EMAIL = "xxx@example.com"  # replace with your real email
MAX_QUESTIONS = 5


load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
print("HF_TOKEN:", os.getenv("HF_TOKEN"))

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
print("Status code is", response.status_code)

from huggingface_hub import HfApi

api = HfApi()
token_info = api.whoami(token=HF_TOKEN)
if "scope" in token_info:
    scopes = token_info["scope"]
    print(scopes)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]


def chunk_text(text, max_length=500):  
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start = end
    return chunks


def load_chunks_and_embeddings(resume_text, github_text):
    with open("me/summary2.txt", "r", encoding="utf-8") as f:
        summary_text = f.read()

    combined_text = resume_text + "\n\n" + summary_text + "\n\n" + github_text
    chunks = chunk_text(combined_text)

    with open("me/embeddings.json", "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    return chunks, embeddings



def find_similar_chunks(question_embedding, chunks, embeddings, top_k=3):
    embeddings_np = np.array(embeddings)
    question_emb_np = np.array(question_embedding).reshape(1, -1)
    similarities = cosine_similarity(question_emb_np, embeddings_np)[0]

    top_k = min(top_k, len(chunks))  # <-- prevent out of range error

    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

class Me:

    def __init__(self, chunks, embeddings):
        self.openai = OpenAI()
        self.name = "Pooja Nigam"
        self.chunks = chunks
        self.embeddings = embeddings

        reader = PdfReader("me/Pooja_Nigam_Resume.pdf")
        self.resume = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.resume += text
        with open("me/summary2.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
        with open("me/github_profile.txt", "r", encoding="utf-8") as f:
            self.github_profile = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self, context=""):
        system_prompt = f"""You are acting as {self.name}. 
        You answer questions about {self.name}'s professional background, skills, experience, and projects, drawing from their resume, 
        background summary, and GitHub profile. Your goal is to represent {self.name} honestly and naturally — be professional, 
        but conversational and approachable, as if speaking with an HR recruiter or a potential employer. Avoid robotic language and 
        canned phrases and repetitive closing phrases like like "feel free to ask more" or "let me know if you want to know anything else.", etc. 
        Keep answers clear, direct, and original. If you don’t know the answer to a question, use the record_unknown_question tool 
        to log it, even if it’s unrelated to career. If the conversation moves toward networking or follow-up, ask for their email and store it with the record_user_details tool.
        ## Summary:
        {self.summary}

        ## Resume:
        {self.resume}

        ## GitHub Profile:
        {self.github_profile}

         ## Relevant context from resume and summary:
        {context}

        With this information, chat with the user as {self.name}, using a natural, confident, and engaging tone."""
    
        return system_prompt

    def chat(self, message, history, user_email=None):
        if user_email != ADMIN_EMAIL:  # admin bypass
            user = get_user(user_email)
            if user:
                _, questions_asked = user
                if questions_asked >= MAX_QUESTIONS:
                    return f"You have reached the {MAX_QUESTIONS}-question limit."
                else:
                    increment_questions(user_email)
            else:
            # add new user
                add_user(user_email)
                increment_questions(user_email)

        user_message = message  # store original user text

    # 1. Check if question is already answered in DB
        answer = get_answer(user_message)
        if answer:
            return answer  # Return cached answer immediately

    # 2. Embed question for RAG retrieval
        question_embedding_response = self.openai.embeddings.create(model="text-embedding-3-small",input=user_message)
        question_embedding = question_embedding_response.data[0].embedding

        relevant_chunks = find_similar_chunks(question_embedding, self.chunks, self.embeddings, top_k=3)
        context = "\n\n".join(relevant_chunks)

    # 3. Prepare system prompt with retrieved context
        system_prompt = self.system_prompt(context=context)

        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_message}]

        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason == "tool_calls":
                tool_message = response.choices[0].message
                tool_calls = tool_message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(tool_message)
                messages.extend(results)
            else:
                done = True

        final_answer = str(response.choices[0].message.content)

    # 4. Save Q&A to DB 
        add_qa(user_message, final_answer)

    # 5. If unknown answer, log it
        if "I don't know" in final_answer or "Sorry" in final_answer:
            add_unknown_question(user_message)

        return final_answer


if __name__ == "__main__":
    reader = PdfReader("me/Pooja_Nigam_Resume.pdf")
    resume_text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])

    with open("me/github_profile.txt", "r", encoding="utf-8") as f:
        github_text = f.read()
    
    chunks, embeddings = load_chunks_and_embeddings(resume_text, github_text)
    me = Me(chunks, embeddings)
    gr.ChatInterface(me.chat, type="messages").launch()

