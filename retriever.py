import argparse
import os
from dotenv import load_dotenv

# from langchain_community.vectorstores import Chroma   #will be deprecated in future
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI

load_dotenv()

# === CONFIG ===
CHROMA_PATH = "chroma_db"
MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai or openrouter
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

PROMPT_TEMPLATE = """
You are an expert assistant. Answer the user's question based only on the following context:

{context}

---

Question: {question}
"""

# === RETRIEVAL + ANSWERING ===
def retrieve_and_respond(query_text):
    # Load vector DB
    print("[🔍] Loading Chroma vector store...")
    embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    print(f"[🔎] Searching for: \"{query_text}\"")
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if not results or results[0][1] < 0.7:
        print("[⚠️] No good matches found.")
        return

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    sources = [doc.metadata.get("source", "unknown") for doc, _ in results]

    # Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_text)

    # Get answer
    if MODEL_PROVIDER == "openrouter":
        print("[🤖] Using OpenRouter (GPT-4o Mini)...")
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
    else:
        print("[🤖] Using OpenAI...")
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18", temperature=0)
        response = model.invoke(prompt)   #predict ki jgh invoke for openai taake future mai deprecate na hw
        response_text = response.content  # ✅ Extract the assistant's message

    print("\n[✅] Final Answer:\n")
    print(response_text)
    print("\n📎 Sources:\n")
    for src in sources:
        print(f"- {src}")

# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Ask questions from the embedded vector store.")
    parser.add_argument("query_text", type=str, help="The user query.")
    args = parser.parse_args()

    retrieve_and_respond(args.query_text)

if __name__ == "__main__":
    main()
