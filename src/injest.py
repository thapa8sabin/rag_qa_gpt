import os
import glob
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "quiz_knowledge_base")
DOCS_DIR = "data"

def main():
    # 1. Ensure target directory exists and find target files (.md and .txt)
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"📁 Created empty '{DOCS_DIR}/' directory. Please drop your files there and re-run.")
        return

    # Look for both Markdown and Text files in the directory
    supported_extensions = ["*.md", "*.txt"]
    file_paths = []
    for ext in supported_extensions:
        file_paths.extend(glob.glob(os.path.join(DOCS_DIR, ext)))

    if not file_paths:
        print(f"⚠️ No files (.md or .txt) found inside the '{DOCS_DIR}/' folder.")
        return

    print(f"📚 Found {len(file_paths)} document(s) to process.")

    # 2. Configure Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["--------------------------------------------------------------------------------"],
        keep_separator=False
    )

    # 3. Load local embedding model
    print("🧠 Loading local embedding model (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 4. Initialize Qdrant Client
    client = QdrantClient(url=QDRANT_URL)
    
    # 5. Create Collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"📦 Created new Qdrant collection: '{COLLECTION_NAME}'")

    points = []
    global_point_id = 0

    # 6. Loop through each file, split, encode, and prepare points
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"📖 Processing: {filename}...")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}. Skipping.")
            continue

        # Split current file text into chunks
        chunks = text_splitter.split_text(content)
        print(f"   ↳ Split into {len(chunks)} chunks.")

        # Vectorize chunks and assign metadata
        for chunk in chunks:
            vector = encoder.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id=global_point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_file": filename  # Added metadata tracker
                    }
                )
            )
            global_point_id += 1
        
    # 7. Batch Upsert everything into Qdrant
    if points:
        print(f"🚀 Uploading {len(points)} total vectors to Qdrant...")
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print("✅ Multi-document ingestion completed successfully!")
    else:
        print("❌ No valid vectors generated to upload.")

if __name__ == "__main__":
    main()