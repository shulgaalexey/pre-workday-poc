import json
import pathlib

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agent import _get_vector_memory

try:
    from .openai_config import get_openai_api_key
except ImportError:
    # Handle case when module is run directly
    from openai_config import get_openai_api_key

emb = OpenAIEmbeddings(openai_api_key=get_openai_api_key())

# Collect all translation pairs first
texts = []
with open("data/seed_tm.json") as f:
    for row in json.load(f):
        for lang, tgt in row.items():
            if lang not in {"src"}:
                texts.append(f"{row['src']} -> {tgt}")

# Create FAISS index with all texts at once
if texts:
    vs = FAISS.from_texts(texts, emb)
    vs.save_local("translation_mem.index")
else:
    print("No translation pairs found in seed data")
    exit(1)
print("Seeded TM completed")
