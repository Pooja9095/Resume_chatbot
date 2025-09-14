# Career Conversations – Resume Chatbot

Interact with a smart chatbot that knows my professional background, skills, experience, and career details! Ask it questions about my resume, GitHub profile, and career details, and it responds intelligently using a combination of database storage, embeddings, and retrieval techniques.  

Check it out live here: [Career Conversations on Hugging Face Spaces](https://huggingface.co/spaces/Pooja-Nigam/Career_Conversation)

---

## Features

- **Resume & GitHub Awareness**: The bot answers questions based on my resume PDF, GitHub profile text, and a background summary.  
- **RAG (Retrieval-Augmented Generation)**: Uses semantic search to fetch relevant context from my uploaded documents before generating an answer.  
- **SQLite Database**:  
  - Stores questions & answers for instant responses to repeated queries.
  - Tracks **per-user session IDs** and limits free users to 5 messages per session.  
  - Logs unknown questions for future learning.  
- **Push Notifications**: Integrates Pushover to notify me whenever a user asks an unknown question.  
- **Admin Control**: Special email bypass for unlimited questions.  
- **Embeddings**: Uses OpenAI embeddings to semantically match questions to my documents.
- **Flexible APIs**: Can use any free or paid API for language models, e.g., OpenAI, Groq, Llama, etc.    

---

## Tools & Libraries Used

- Python 3.x  
- [Gradio](https://gradio.app/) – for the interactive web interface  
- [Mistral API](https://docs.mistral.ai/) – embeddings and Mistral AI responses  
- [Hugging Face Hub](https://huggingface.co/) – to deploy the Space publicly  
- [PyPDF](https://pypi.org/project/pypdf/) – to extract text from PDF resumes  
- [Requests](https://pypi.org/project/requests/) – API requests for push notifications and HF API  
- [SQLite](https://www.sqlite.org/index.html) – database for Q&A, users, and unknown questions  
- [Pushover](https://pushover.net/) – push notifications for tracking unknown questions and user details  
- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) – for cosine similarity in RAG  

---

## Usage

1. Clone the repository:  
```bash
git clone https://github.com/Pooja9095/Resume_chatbot.git
cd Resume_chatbot
```
2. Install requirements:  
```bash
pip install -r requirements.txt
```
3. Create a .env file with your keys (these are ignored in Git for security):
```bash
HF_TOKEN=your_huggingface_token
MISTRAL_API_KEY=your_mistral_key # or any other API key
PUSHOVER_TOKEN=your_pushover_token
PUSHOVER_USER=your_pushover_user_key
```
4. Run the chatbot locally:
```bash
python resume_bot.py
```
## License

MIT License  

Copyright (c) 2025 Pooja Nigam  

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

