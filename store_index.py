from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

def store_chunks_in_pinecone(text_chunks, batch_size=50):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "medical-chatbot"

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    texts = [doc.page_content for doc in text_chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32).tolist()

    vectors = [
        {
            "id": f"doc-{i}",
            "values": emb,
            "metadata": {"text": texts[i][:1000]}  # truncate for token control
        }
        for i, emb in enumerate(embeddings)
    ]

    index = pc.Index(index_name)
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

    print(f"Stored {len(vectors)} chunks to Pinecone index '{index_name}'")
