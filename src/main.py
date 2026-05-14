import requests
import os
from dotenv import load_dotenv
from utils import load_data
from rag import RAGEngine

# -------------------------------
#Load API Key
# -------------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# -------------------------------
#Base Directory Setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------
#Load Dataset
# -------------------------------
data_path = os.path.join(BASE_DIR, "data", "training_data.json")
data = load_data(data_path)

# -------------------------------
#Initialize RAG Engine
# -------------------------------
rag_engine = RAGEngine(data)

# -------------------------------
#Load Prompt Template
# -------------------------------
template_path = os.path.join(BASE_DIR, "prompts", "template.txt")

with open(template_path, "r", encoding="utf-8") as f:
    template = f.read()


# -------------------------------
#Query Function
# -------------------------------
def query_model(user_input):
    #Retrieve relevant examples using RAG
    examples = rag_engine.retrieve(user_input, top_k=3)

    prompt = template.format(
        examples=examples,
        user_input=user_input
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return response.json()


# -------------------------------
#Run Script
# -------------------------------
if __name__ == "__main__":
    user_question = input("Enter your question: ")
    result = query_model(user_question)

    print("\n--- MODEL OUTPUT ---\n")

    try:
        print(result["choices"][0]["message"]["content"])
    except Exception:
        print("Error:", result)