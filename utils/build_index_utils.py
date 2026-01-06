from typing import List, Dict
import uuid
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def load_wiki_documents(knowledge_json_path: str) -> List[Dict]:
    """Load Wikipedia documents from JSON file"""
    import json

    logger.info(f"Loading documents from {knowledge_json_path}")
    with open(knowledge_json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def prepare_metadata(title: str) -> Dict:
    """Prepare metadata for a document"""
    return {"title": title}