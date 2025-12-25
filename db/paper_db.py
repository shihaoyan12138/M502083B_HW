import chromadb
from chromadb.config import Settings
from pathlib import Path


class PaperVectorDB:
    def __init__(self, persist_dir="./index/paper_db"):
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir)
        )

        self.collection = self.client.get_or_create_collection(
            name="papers"
        )
        print("Paper DB dir:", persist_dir.resolve())

    def add_paper(self, paper_id, text, embedding, metadata):
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[paper_id]
        )
        # self.client.persist()

    def search(self, query_embedding, top_k=3):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
