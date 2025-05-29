import os
import requests
from io import BytesIO
import tempfile
import shutil
from typing import Generator, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings  #deprecated
# from langchain.vectorstores import Chroma          #deprecated
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma    #will be deprecated in near future
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI

# === Load ENV ===
from dotenv import load_dotenv
load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
# print(TENANT_ID)
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SHAREPOINT_SITE = os.getenv("SITE_URL_NEW")  # https://tenant.sharepoint.com/sites/your-site
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Step 1: Get Access Token ===
def get_access_token():
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default"
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

# === Step 2: Get All Files (Recursively) ===
def fetch_files(token: str) -> Generator[Tuple[str, BytesIO], None, None]:
    headers = {"Authorization": f"Bearer {token}"}
    site_name = SHAREPOINT_SITE.split("/")[-1]
    
    # Get site ID
    site_res = requests.get(f"https://graph.microsoft.com/v1.0/sites/root:/sites/{site_name}", headers=headers)
    site_res.raise_for_status()
    site_id = site_res.json()["id"]

    # Get drive ID
    drive_res = requests.get(f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive", headers=headers)
    drive_res.raise_for_status()
    drive_id = drive_res.json()["id"]

    # Recursive folder traversal
    def traverse_items(folder_id="root"):
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children"
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        for item in res.json().get("value", []):
            if item.get("folder"):
                yield from traverse_items(item["id"])
            elif item.get("file"):
                file_name = item["name"]
                file_id = item["id"]
                download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"
                f = requests.get(download_url, headers=headers)
                f.raise_for_status()
                yield file_name, BytesIO(f.content)

    yield from traverse_items()

def save_temp_file(file_stream, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_stream.read())
    tmp.flush()
    return tmp.name

def load_document(file_name, file_stream):
    ext = os.path.splitext(file_name)[1].lower()
    print(f"[📂 Loading] File: {file_name}, Extension: {ext}")
    
    from langchain_community.document_loaders import (
        PyPDFLoader, UnstructuredExcelLoader,
        UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,
        CSVLoader, TextLoader
    )

    try:
        if ext == ".pdf":
            path = save_temp_file(file_stream, ".pdf")
            docs = PyPDFLoader(path).load()
            for d in docs:
                d.metadata["source"] = file_name
            print(f"[✅ PDF Loaded] Pages: {len(docs)}")
            return docs

        elif ext == ".docx":
            path = save_temp_file(file_stream, ".docx")
            docs = UnstructuredWordDocumentLoader(path).load()
            for d in docs:
                d.metadata["source"] = file_name
            print(f"[✅ DOCX Loaded] Pages: {len(docs)}")
            return docs

        elif ext == ".pptx":
            path = save_temp_file(file_stream, ".pptx")
            docs = UnstructuredPowerPointLoader(path).load()
            for d in docs:
                d.metadata["source"] = file_name
            print(f"[✅ PPTX Loaded] Slides: {len(docs)}")
            return docs

        elif ext in [".xls", ".xlsx"]:
            path = save_temp_file(file_stream, ".xlsx")
            docs = UnstructuredExcelLoader(path).load()
            for d in docs:
                d.metadata["source"] = file_name
            print(f"[✅ Excel Loaded] Sheets: {len(docs)}")
            return docs

        elif ext == ".csv":
            path = save_temp_file(file_stream, ".csv")
            docs = CSVLoader(path).load()
            for d in docs:
                d.metadata["source"] = file_name
            print(f"[✅ CSV Loaded] Rows: {len(docs)}")
            return docs

        elif ext == ".txt":
            path = save_temp_file(file_stream, ".txt")
            docs = TextLoader(path).load()
            for d in docs:
                d.metadata["source"] = file_name
            print(f"[✅ TXT Loaded] Lines: {len(docs)}")
            return docs

        elif ext in [".mp3", ".mp4"]:
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                temp_path = save_temp_file(file_stream, ".mp3")

                with open(temp_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f
                    )

                print(f"[✅ MP3 Transcribed] Length: {len(transcription.text)} chars")
                return [Document(page_content=transcription.text, metadata={"source": file_name})]
            except Exception as e:
                print(f"[⚠️ MP3 Transcription ERROR] {file_name}: {e}")
                return []

        else:
            print(f"[ℹ️ Unsupported file type] Skipping: {file_name}")
            return []

    except Exception as e:
        print(f"[❌ ERROR loading {file_name}]: {e}")
        return []

# === Step 4: Chunking ===
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# === Step 5: Embedding and Storing ===
def embed_and_store(chunks):
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import shutil

    persist_path = "chroma_db"
    if os.path.exists(persist_path):
        print(f"[♻️ Removing Existing Vector Store] {persist_path}")
        shutil.rmtree(persist_path)

    print("[💡 Creating Embedding Model]")
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print(f"[💾 Creating New Vector Store] Chunks: {len(chunks)}")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_path)
    print("[✅ Vector Store Persisted to Disk]")

# === RUN ===
def main():
    token = get_access_token()
    print("[🔑 Access Token Retrieved]")

    all_chunks = []
    for file_name, file_stream in fetch_files(token):
        print(f"\n[📥 Processing File] {file_name}")
        docs = load_document(file_name, file_stream)
        if not docs:
            print(f"[⚠️ No Documents Returned] Skipping {file_name}")
            continue
        chunks = chunk_documents(docs)
        print(f"[✂️ Chunked] {file_name}: {len(chunks)} chunks")
        # Show the first chunk and its metadata
        if chunks:
            print(f"[🧾 First Chunk Preview] {chunks[0].page_content[:300]}...")
            print(f"[📎 Metadata] {chunks[0].metadata}")
        all_chunks.extend(chunks)

    if all_chunks:
        print(f"\n[💾 Storing Embeddings] Total chunks: {len(all_chunks)}")
        embed_and_store(all_chunks)
        print("[✅ Done] Embeddings stored in ChromaDB.")
    else:
        print("[⚠️ No chunks created. Nothing stored.]")

if __name__ == "__main__":
    main()