import logging
import uuid
from typing import List, Dict

import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel

from config.build_index_config import *
from utils.build_index_utils import (
    normalize_embeddings,
    load_wiki_documents,
    prepare_metadata
)

logger = logging.getLogger(__name__)


class ChromaDBIndexer:

    def __init__(
            self,
            chroma_db_path: str = CHROMA_DB_PATH,
            clip_model_name: str = CLIP_MODEL_NAME,
            clip_collection_name: str = CLIP_COLLECTION_NAME,
            recreate_collections: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize models
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {chroma_db_path}")
        self.client = chromadb.PersistentClient(path=chroma_db_path)

        if recreate_collections:
            self.client.delete_collection(CLIP_COLLECTION_NAME)

        self.clip_collection = self.client.get_or_create_collection(
            name=clip_collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def encode_with_clip(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using CLIP model"""
        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            clip_embeddings = self.clip_model.get_text_features(**inputs).cpu().numpy()

        clip_embeddings = normalize_embeddings(clip_embeddings).tolist()
        return clip_embeddings

    def add_batch(
            self,
            titles: List[str],
            contents: List[str],
            ids: List[str],
            metadata: List[Dict]
    ):
        clip_embeddings = self.encode_with_clip(titles)
        self.clip_collection.add(
            ids=ids,
            embeddings=clip_embeddings,
            documents=contents,
            metadatas=metadata
        )

    def build_index_from_json(
            self,
            wiki_json_path: str,
            batch_size: int = BATCH_SIZE,
    ):
        """Build index from Wikipedia JSON file"""
        documents = load_wiki_documents(wiki_json_path)

        batch_titles = []
        batch_contents = []
        batch_ids = []
        batch_metadata = []

        total_processed = 0

        for document in documents:
            title = document['title']
            content = document['content']

            batch_titles.append(title)
            batch_contents.append(content)
            batch_ids.append(str(uuid.uuid4()))
            batch_metadata.append(prepare_metadata(title))

            if len(batch_titles) == batch_size:
                self.add_batch(batch_titles, batch_contents, batch_ids, batch_metadata)

                total_processed += len(batch_titles)

                # Clear batch
                batch_titles = []
                batch_contents = []
                batch_ids = []
                batch_metadata = []

        # Process remaining documents
        if batch_titles:
            self.add_batch(batch_titles, batch_contents, batch_ids, batch_metadata)
            total_processed += len(batch_titles)

        logger.info(f"Finished! Total processed: {total_processed} documents")
        return total_processed