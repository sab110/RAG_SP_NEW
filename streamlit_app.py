import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
Use the following context to answer the user's question.

Context:
{context}

---

User: {question}
Assistant:"""

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page config and title
st.set_page_config(page_title="AI Chat (Docs)", layout="centered")
st.title("🧠 AI Chat Assistant (SharePoint Docs)")

# Display existing messages
for msg, role in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Input box
user_input = st.chat_input("Ask your question...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append((user_input, "user"))

    # Search vector store
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    with st.status("🔍 Searching the vector store...", expanded=False):
        results = db.similarity_search_with_relevance_scores(user_input, k=5)

    if not results or results[0][1] < 0.7:
        bot_response = "❌ I couldn't find relevant information for that."
    else:
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        sources = list({doc.metadata.get("source", "unknown") for doc, _ in results})
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context, question=user_input
        )

        # LLM response with loading indicator
        with st.status("🤖 Generating response...", expanded=False) as status:
            if LLM_PROVIDER == "openrouter":
                client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                bot_response = response.choices[0].message.content
            else:
                llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18", temperature=0)
                response = llm.invoke(prompt)
                bot_response = response.content

            # Append sources to the assistant's message
            if sources:
                formatted_sources = "\n\n📎 **Sources:**\n" + "\n".join([f"- `{src}`" for src in sources])
                bot_response += formatted_sources

            status.update(label="✅ Done", state="complete")

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.chat_history.append((bot_response, "assistant"))
