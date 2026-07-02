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
Answer using the provided context.

If the answer is partially available, summarize the available information.

Only say "I couldn't find..." if the retrieved context contains no relevant information.
Context:
{context_text}

Question:
{question}

Answer:
"""
    
    response= model.generate_content(prompt)
    return response.text