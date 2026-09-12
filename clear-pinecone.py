import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index(os.environ["INDEX_NAME"])

index.delete(delete_all=True)

print("All records deleted.")