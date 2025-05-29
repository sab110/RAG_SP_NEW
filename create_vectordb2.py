# source contains sharepoint doc url

import os, requests, shutil, tempfile
from io import BytesIO
from typing import Generator, Tuple, List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SHAREPOINT_SITE = os.getenv("SITE_URL_NEW")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def fetch_files(token: str) -> Generator[Tuple[str, BytesIO, str], None, None]:
    headers = {"Authorization": f"Bearer {token}"}
    site_name = SHAREPOINT_SITE.split("/")[-1]

    site_res = requests.get(f"https://graph.microsoft.com/v1.0/sites/root:/sites/{site_name}", headers=headers)
    site_res.raise_for_status()
    site_id = site_res.json()["id"]

    drive_res = requests.get(f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive", headers=headers)
    drive_res.raise_for_status()
    drive_id = drive_res.json()["id"]

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
                web_url = item.get("webUrl", f"https://sharepoint.com/{file_name}")
                content_res = requests.get(
                    f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content", headers=headers)
                content_res.raise_for_status()
                yield file_name, BytesIO(content_res.content), web_url

    yield from traverse_items()

def save_temp_file(file_stream, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_stream.read())
    tmp.flush()
    return tmp.name

def load_document(file_name, file_stream, url: str) -> List[Document]:
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
        elif ext == ".docx":
            path = save_temp_file(file_stream, ".docx")
            docs = UnstructuredWordDocumentLoader(path).load()
        elif ext == ".pptx":
            path = save_temp_file(file_stream, ".pptx")
            docs = UnstructuredPowerPointLoader(path).load()
        elif ext in [".xls", ".xlsx"]:
            path = save_temp_file(file_stream, ".xlsx")
            docs = UnstructuredExcelLoader(path).load()
        elif ext == ".csv":
            path = save_temp_file(file_stream, ".csv")
            docs = CSVLoader(path).load()
        elif ext == ".txt":
            path = save_temp_file(file_stream, ".txt")
            docs = TextLoader(path).load()
        elif ext in [".mp3", ".mp4"]:
            temp_path = save_temp_file(file_stream, ext)
            client = OpenAI(api_key=OPENAI_API_KEY)
            with open(temp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
            docs = [Document(page_content=transcription.text)]
        else:
            print(f"[ℹ️ Unsupported file type] Skipping: {file_name}")
            return []

        for d in docs:
            d.metadata["source"] = url
        print(f"[✅ Loaded] {file_name} -> {len(docs)} document(s)")
        return docs

    except Exception as e:
        print(f"[❌ ERROR loading {file_name}]: {e}")
        return []

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def embed_and_store(chunks):
    persist_path = "chroma_db"
    if os.path.exists(persist_path):
        print(f"[♻️ Removing Existing Vector Store] {persist_path}")
        shutil.rmtree(persist_path)

    print("[💡 Creating Embedding Model]")
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    print(f"[💾 Creating New Vector Store] Chunks: {len(chunks)}")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_path)
    print("[✅ Vector Store Persisted to Disk]")

def main():
    token = get_access_token()
    print("[🔑 Access Token Retrieved]")

    all_chunks = []
    for file_name, file_stream, file_url in fetch_files(token):
        print(f"\n[📥 Processing File] {file_name}")
        docs = load_document(file_name, file_stream, file_url)
        if not docs:
            print(f"[⚠️ No Documents Returned] Skipping {file_name}")
            continue
        chunks = chunk_documents(docs)
        print(f"[✂️ Chunked] {file_name}: {len(chunks)} chunks")
        if chunks:
            print(f"[🧾 First Chunk Preview] {chunks[0].page_content[:300]}...")
            print(f"[📎 First Metadata] {chunks[0].metadata}")
        all_chunks.extend(chunks)

    if all_chunks:
        print(f"\n[💾 Storing Embeddings] Total chunks: {len(all_chunks)}")
        embed_and_store(all_chunks)
        print("[✅ Done] Embeddings stored in ChromaDB.")
    else:
        print("[⚠️ No chunks created. Nothing stored.]")

if __name__ == "__main__":
    main()
