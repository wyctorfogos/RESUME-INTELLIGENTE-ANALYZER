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

def recommend_with_chat_ollama(query: str, matches: List[Dict[str, Any]], ollama_model_name:str="gemma3:270m", chat_ollama_template:str=None) -> str:
    context = "\n\n".join(
        f"chunk_id={item['chunk_id']} score={item['score']:.4f}\n{item['content'][:1200]}"
        for item in matches
    )

    try:
        llm = ChatOllama(model=ollama_model_name, temperature=0.3)
        prompt = ChatPromptTemplate.from_template(chat_ollama_template)
        response = (prompt | llm).invoke({"query": query, "context": context})
        return str(response.content)
    except Exception as exc:
        return (
            "ChatOllama unavailable. Returning template + retrieved context.\n\n"
            f"Reason: {exc}\n\n"
            + chat_ollama_template.format(query=query, context=context)
        )
