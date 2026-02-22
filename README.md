# Resume Recommendation com FAISS + OLLAMA

Sistema de recomendação de vagas baseado em **RAG (Retrieval-Augmented Generation)**.
O projeto indexa descrições de vagas em um banco vetorial com **FAISS** e usa **OLLAMA** para gerar recomendações contextuais a partir de uma query de perfil/candidato.

## Como funciona

1. O dataset de vagas (`data/archive/resume_data.csv`) é carregado.
2. As colunas mais relevantes da vaga são agrupadas em chunks (`vector_dataset`).
3. Cada chunk é transformado em embedding com `sentence-transformers`.
4. Os embeddings são indexados em FAISS para busca por similaridade.
5. Uma query do candidato recupera os `top_k` chunks mais próximos.
6. O contexto recuperado é enviado para um modelo local via OLLAMA para recomendação final.

## Stack principal

- Python
- FAISS (`faiss-cpu`)
- Sentence Transformers
- LangChain Community (`ChatOllama`)
- Pandas / NumPy

## Estrutura relevante

- `src/scripts/rag.py`: pipeline principal de ingestão, indexação e recomendação.
- `src/scripts/models/FaissDatabaseClass.py`: funções de criação/consulta do banco vetorial.
- `src/scripts/models/LLMRecommender.py`: integração com OLLAMA para resposta final.
- `config/.env`: variáveis de configuração.

## Configuração

1. Crie e ative seu ambiente virtual.
2. Instale dependências:

```bash
pip install -r requirements.txt
pip install sentence-transformers langchain-core python-dotenv
```

3. Configure `config/.env` (exemplo):

```env
CHAT_OLLAMA_TEMPLATE="You are a job recommendation assistant.Use only the context below to recommend the best chunks for the query.\n\nQuery:\n{query}\n\nContext:\n{context}\nReturn:\n1) top chunks ordered by relevance\n2) why each chunk is relevant\n3) a short final recommendation\n"
ollama_model_name="qwen3:0.6b"
```

4. Garanta que o OLLAMA está ativo e o modelo foi baixado:

```bash
ollama pull qwen3:0.6b
```

## Execução

Execute o pipeline:

```bash
python src/scripts/rag.py
```

Também é possível sobrescrever parâmetros por variável de ambiente no comando:

```bash
RAG_QUERY="Python, NLP e Machine Learning" \
RAG_TOP_K=5 \
RAG_BATCH_SIZE=32 \
python src/scripts/rag.py
```

## Artefatos gerados

Após execução, os arquivos abaixo são atualizados em `data/vector_dataset/`:

- `vector_dataset.json`: chunks textuais com metadados.
- `embeddings.npy`: embeddings dos chunks.
- `faiss.index`: índice vetorial FAISS.

## Resultado esperado

No terminal, o script imprime:

- `Top chunks` recuperados por similaridade (com score e intervalo de linhas).
- `Recommendation` gerada pelo OLLAMA com base no contexto recuperado.
