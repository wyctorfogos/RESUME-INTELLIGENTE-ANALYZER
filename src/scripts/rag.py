import json
import os
from typing import Any, Dict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from models.FaissDatabaseClass import *
from models.LLMRecommender  import *
from dotenv import load_dotenv

load_dotenv("./config/.env")

if __name__ == "__main__":
    batch_size = int(os.getenv("RAG_BATCH_SIZE", "32"))
    top_k = int(os.getenv("RAG_TOP_K", "5"))
    csv_path = os.getenv("RAG_DATASET_PATH", "./data/archive/resume_data.csv")
    embeddings_model = os.getenv("RAG_EMBEDDINGS_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    query = os.getenv("RAG_QUERY", "I'm a python machine learning data scientist developper with a Master 2 in Data Science - Computer Vision")

    TARGET_COLUMNS = [
        "job_position_name",
        "educationaL_requirements",
        "experiencere_requirement",
        "age_requirement",
        "responsibilities.1",
        "skills_required",
    ]
    ollama_model_name=os.getenv('ollama_model_name')
    CHAT_OLLAMA_TEMPLATE = os.getenv('CHAT_OLLAMA_TEMPLATE')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, sep=",")
    df = normalize_job_column(df)

    missing_columns = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    filtered_df = df[TARGET_COLUMNS].copy()
    vector_dataset = create_vector_dataset(filtered_df, batch_size=batch_size, targets_columns=TARGET_COLUMNS)

    if not vector_dataset:
        raise ValueError("vector_dataset is empty. Check input dataset or batch size.")

    model = SentenceTransformer(embeddings_model)
    chunk_texts = [item["content"] for item in vector_dataset]
    embeddings = model.encode(chunk_texts, convert_to_numpy=True).astype("float32")

    index = create_faiss_index(embeddings=embeddings)
    matches = retrieve_chunks(
        query=query,
        model=model,
        index=index,
        vector_dataset=vector_dataset,
        top_k=top_k,
    )

    os.makedirs("./data/vector_dataset", exist_ok=True)
    with open("./data/vector_dataset/vector_dataset.json", "w", encoding="utf-8") as f:
        json.dump(vector_dataset, f, ensure_ascii=False, indent=2)
    np.save("./data/vector_dataset/embeddings.npy", embeddings)
    faiss.write_index(index, "./data/vector_dataset/faiss.index")

    recommendation = recommend_with_chat_ollama(query=query, matches=matches, ollama_model_name=ollama_model_name, chat_ollama_template=CHAT_OLLAMA_TEMPLATE)
    print("Top chunks:")
    for item in matches:
        print(
            f"- chunk_id={item['chunk_id']} score={item['score']:.4f} rows={item['row_start']}..{item['row_end']}"
        )
    print("\nRecommendation:\n")
    print(recommendation)

    
