import json
import os
from typing import Any, Dict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv("./config/.env")


def normalize_job_column(df: pd.DataFrame) -> pd.DataFrame:
    if "\ufeffjob_position_name" in df.columns and "job_position_name" not in df.columns:
        df = df.rename(columns={"\ufeffjob_position_name": "job_position_name"})
    return df


def build_chunk_text(record: pd.Series, targets_columns:list) -> str:
    parts = []
    for col in targets_columns:
        value = record.get(col, "")
        if pd.notna(value) and str(value).strip():
            parts.append(f"{col}: {str(value).strip()}")
    return "\n".join(parts)


def create_vector_dataset(df: pd.DataFrame, targets_columns:list, batch_size: int=32) -> List[Dict[str, Any]]:
    vector_dataset: List[Dict[str, Any]] = []
    for chunk_id, start in enumerate(range(0, len(df), batch_size)):
        batch = df.iloc[start : start + batch_size]
        chunk_texts = [build_chunk_text(row, targets_columns=targets_columns) for _, row in batch.iterrows()]
        chunk_texts = [txt for txt in chunk_texts if txt]
        if not chunk_texts:
            continue

        vector_dataset.append(
            {
                "chunk_id": chunk_id,
                "row_start": int(start),
                "row_end": int(start + len(batch) - 1),
                "content": "\n\n---\n\n".join(chunk_texts),
            }
        )
    return vector_dataset


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_chunks(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    vector_dataset: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    matches: List[Dict[str, Any]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(vector_dataset):
            continue
        entry = dict(vector_dataset[idx])
        entry["score"] = float(score)
        matches.append(entry)
    return matches
