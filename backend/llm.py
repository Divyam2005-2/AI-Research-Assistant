import os
import google.generativeai as genai
from dotenv import load_dotenv

loaded = load_dotenv()
print("Dotenv loaded:", loaded)
print("API KEY:", os.getenv("GEMINI_API_KEY"))
genai.configure(
    api_key = os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(question, context):
    context_text = "\n\n".join(context)

    prompt = f"""
You are an AI Research Assistant.

Your task is to answer the user's question ONLY using the information provided in the context below.

Rules:
1. Use only the provided context.
2. Do not make up or assume information.
3. If the answer is not available in the context, reply:
   "I couldn't find that information in the uploaded document."
4. Keep the answer clear, concise, and easy to understand.

Context:
{context_text}

Question:
{question}

Answer:
"""
    
    response= model.generate_content(prompt)
    return response.text