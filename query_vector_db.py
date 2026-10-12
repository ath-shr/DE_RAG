import os
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()

embedding_model = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b", api_key=os.getenv("NVIDIA_API_KEY"))



persistent_directory = "db/chroma_db"

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

#get all chunk IDs from the vector store
data = db.get()
chunk_ids = data["ids"]

print(f"Total chunks: {len(chunk_ids)}")
print("Chunk IDs:", chunk_ids)
# Chunk IDs: ['80836d72-1974-4ab8-acac-8f66b19014d1', 'fb9dcfff-626e-4442-8bae-1b5af62b3c67', 'e51196e3-db4c-4a2b-80cb-e074a0e5908e', '3303727c-a57e-427b-8a15-ddf6df966be1', '13f3602f-6d48-46da-b848-b80c2ca576c3', 'dd5e13be-3db9-4c57-b871-705f0aff96c7', 'd64e1cfe-4981-4cf5-a9f0-8bf4de052bb8', 'cdbaf952-2e25-4fa8-bb54-ae7e4120f0e1', 'b0e21f19-d6f8-4d2d-a279-a0cb88ce60dd', 'a11f49bb-3055-4775-bbb6-c30f997f4d7b', '420a0d39-8785-48aa-81bd-38a7880efd45', '5706495a-059a-40e3-995f-01bd5a35d02b', '70c81a49-c85a-4f16-be9f-89e411ba3235', 'c729e1be-0d25-48b7-bafd-cfe1ab662a95', '639fde90-b435-4961-8a8b-26ec9c789789', '8e1c7ae7-666a-48c2-b85c-e8cb342cf0ae', 'cb8e07e6-73f2-4ce5-9e62-2a7d873dbba8', '71923731-7c50-42d2-a515-c62c0eb79e86']


# get content of a specific chunk by its ID
# Assuming your 'vector_store' is already initialized...
target_id = "80836d72-1974-4ab8-acac-8f66b19014d1"

# Fetch the chunk(s)
chunks = db.get_by_ids([target_id])

if chunks:
    target_chunk = chunks[0]

    print("Page Content:", target_chunk.page_content)
    print("Metadata:", target_chunk.metadata)
else:
    print("Chunk ID not found.")