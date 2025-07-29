import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter



def debug_log(message):
    # if DEBUG:
    print(f"DEBUG: {message}")



#This Below Part is used to update the guidelines

def ensure_vectorstore_exists_and_get():
    base_dir = Path(__file__).resolve().parent
    persist_path = base_dir / "vectorstores" / "chroma_db"

    # If vectorstore is missing or empty, try to upload
    if not os.path.exists(persist_path) or not os.listdir(persist_path):
        debug_log("⚠️ Vectorstore not found or empty. Attempting to upload guideline data...")
        try:
            upload_vector_data()
            debug_log("✅ Vectorstore created.")
        except Exception as e:
            debug_log(f"❌ Failed to create vectorstore: {str(e)}")
            return None

    # Final check — confirm the vectorstore was actually created
    if not os.path.exists(persist_path) or not os.listdir(persist_path):
        debug_log("❌ Vectorstore still not found after upload attempt. Returning None.")
        return None

    # If everything is good, return GuidelinesStore
    return GuidelinesStore()



def upload_vector_data():
    print("🚀 Starting upload to ChromaDB from `data/knowledge_base`")

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "knowledge_base"
    persist_path = base_dir / "vectorstores" / "chroma_db"

    if not data_path.exists():
        print(f"❌ Folder not found: {data_path}")
        raise FileNotFoundError(f"❌ To Upload Data to Vector DB Folder not found: {data_path}")

    # Load all .txt files
    loader = DirectoryLoader(path=str(data_path), glob="**/*.txt", loader_cls=TextLoader)
    raw_docs = loader.load()
    debug_log(f"📄 Loaded {len(raw_docs)} raw text files")

    # Add topic metadata
    for doc in raw_docs:
        filename = os.path.basename(doc.metadata['source'])
        topic = filename.replace(".txt", "")
        doc.metadata.update({"topic": topic})
    debug_log("🏷️  Added 'topic' metadata to all documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    print(f"✂️  Split into {len(chunks)} document chunks")

    # Load Embeddings
    # embeddings = OpenAIEmbeddings()  #or use HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Chroma vectorstore
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path)
    )

    print(f"✅ Vectorstore saved at {persist_path}")


#This Below Part is used to get/retrieve the data


class GuidelinesStore:
    def get_relevant_instructions(self, code_snippet: str, file_path: str) -> List[str]:
        # You can enhance logic here by extracting topic from file_path if needed
        return retrieve_guidelines(file_path=file_path, query=code_snippet)

def retrieve_guidelines(file_path: str="default", query: str = "How to review this code?") -> List[str]:
    topic = infer_topic_from_filename(file_path)
    print(f"🔍 Fetching instructions for: {file_path}  on   📁 Inferred topic: {topic}")
    base_dir = Path(__file__).resolve().parent
    persist_path = base_dir / "vectorstores" / "chroma_db"

    if not persist_path.exists():
        print("❌ Vectorstore not found. Please run `upload_vector_data.py` first.")
        raise FileNotFoundError("Vectorstore not found.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_path)
    )
    print("📦 Loaded Chroma vectorstore")

   #todo if we use qdrant or other vectorstore, we can retrive data based on direct meta data instead of going through all files
    docs = vectorstore.similarity_search(query, k=10)
    debug_log(f"🔎 Retrieved {len(docs)} similar documents")

    if topic == "default":
        print("📝 No specific topic provided, returning top 3 general docs.")
        return [doc.page_content for doc in docs[:3]]

    filtered = [doc.page_content for doc in docs if doc.metadata.get("topic", "").lower() == topic.lower()]
    debug_log(f"🧠 Filtered to {len(filtered)} docs with topic: {topic}")

    if not filtered:
        print("⚠️ No exact topic match found. Returning top 3 general docs")
        return [doc.page_content for doc in docs[:3]]

    return filtered[:5]

def infer_topic_from_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".java":
        return "java_instruction"
    elif ext in [".js", ".jsx", ".tsx"]:
        return "react_instructions"
    elif ext == ".py":
        return "python_instructions"
    elif ext == "general":
        return "general_instructions"
    return "default"

if __name__ == "__main__":
    guidelines_store = None
    if os.environ.get('USE_VECTORSTORE', 'true').lower() == 'true':
        debug_log("Using vectorstore for coding guidelines")
        guidelines_store = ensure_vectorstore_exists_and_get()
    guidelines_text = ""
    normalized_path = "main.java"
    content = "System.out.println(\"Hello World\");"
    if guidelines_store is not None:
        debug_log(f"Getting relevant guidelines for {normalized_path}...")
        relevant_guidelines = guidelines_store.get_relevant_guidelines(
            code_snippet=content,
            file_path=normalized_path
        )
        guidelines_text = "\n".join(relevant_guidelines)
        print(f"Found {len(relevant_guidelines)} relevant guidelines")
    else:
        print("No guidelines available, skipping guidelines context")


