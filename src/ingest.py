import os
from dotenv import load_dotenv
from docx import Document

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()

DOC_PATH = os.getenv("DOC_PATH")
CHROMA_DIR = os.getenv("CHROMA_DIR")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not DOC_PATH or not CHROMA_DIR or not CHROMA_COLLECTION:
    raise ValueError("Missing required env vars")

# ----------------------------
# Load DOCX
# ----------------------------
def load_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = []
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)

print(f"📄 Loading document: {DOC_PATH}")
raw_text = load_docx(DOC_PATH)

# ----------------------------
# Split into chunks
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

chunks = text_splitter.split_text(raw_text)
print(f"✂️  Split into {len(chunks)} chunks")

# ----------------------------
# Embeddings
# ----------------------------
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# ----------------------------
# Chroma (auto-persistent)
# ----------------------------
print(f"🧠 Writing to Chroma at: {CHROMA_DIR}")

vectordb = Chroma(
    collection_name=CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# OPTIONAL: clear existing documents safely
existing = vectordb._collection.count()
if existing > 0:
    print(f"⚠️  Clearing existing {existing} documents")
    vectordb._collection.delete(where={})

vectordb.add_texts(
    texts=chunks,
    metadatas=[{"source": "Desert JetSet Knowledge Base.docx"}] * len(chunks)
)

print("✅ Ingestion complete (auto-persisted)")
