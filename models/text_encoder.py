from sentence_transformers import SentenceTransformer


class TextEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        texts: str or List[str]
        return: embedding(s)
        """
        return self.model.encode(texts, show_progress_bar=False)
