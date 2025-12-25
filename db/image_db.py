import chromadb
from chromadb.config import Settings
from pathlib import Path


class ImageVectorDB:
    def __init__(self, persist_dir="./index/image_db"):
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir)
        )

        self.collection = self.client.get_or_create_collection(
            name="images"
        )

    def add_image(self, image_id, embedding, metadata):
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[image_id]
        )
        # self.client.persist()

    def search(self, query_embedding, top_k=5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
