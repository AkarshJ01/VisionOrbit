import os
import hashlib
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# pyrefly: ignore [missing-import]
from langchain_pinecone import PineconeVectorStore


load_dotenv()


if __name__ == "__main__":
    print("Starting ingestion...")

    # 1. PDF folder

    folder_path = "/Users/akarshjaiswal/Desktop/VisionOrbit/InfoBase"

    # 2. Load all PDFs

    documents = []

    for filename in sorted(os.listdir(folder_path)):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(folder_path, filename)

            print(f"Loading: {filename}")

            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages")

    # 3. Split documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    # 4. Create embeddings

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    # 5. Connect to Pinecone

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    # 6. Generate stable IDs

    ids = []

    for i, document in enumerate(texts):

        source = document.metadata.get("source", "")
        page = document.metadata.get("page", 0)

        unique_string = f"{source}|{page}|{i}|{document.page_content}"

        document_id = hashlib.sha256(
            unique_string.encode("utf-8")
        ).hexdigest()

        ids.append(document_id)

    # 7. Upload in batches

    batch_size = 20

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        print(
            f"Uploading chunks {i + 1} "
            f"to {i + len(batch)}"
        )

        vectorstore.add_documents(
            documents=batch,
            ids=batch_ids
        )

    print("Ingestion complete!")