from typing import List
import logging

import chromadb
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor

from config.generation_config import *
from utils.generate_utils import get_key_frames, flatten_query_results

logger = logging.getLogger(__name__)


class VideoRAGRetriever:
    def __init__(
            self,
            chroma_db_path: str = CHROMA_DB_PATH,
            collection_name: str = CHROMA_COLLECTION_NAME,
            clip_model_name: str = CLIP_MODEL_NAME
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, device_map="auto")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized VideoRAGRetriever with collection: {collection_name}")

    def retrieve(
            self,
            video_path: str,
            n_documents: int = DEFAULT_N_DOCUMENTS,
            frame_threshold: float = DEFAULT_FRAME_THRESHOLD
    ) -> List[str]:
        """
        Retrieve relevant documents from ChromaDB based on video content

        Args:
            question: User question (currently not used in retrieval)
            video_path: Path to video file
            n_documents: Number of documents to retrieve
            frame_threshold: Threshold for keyframe extraction

        Returns:
            List of retrieved document strings
        """
        # Extract keyframes from video
        keyframes = get_key_frames(video_path, frame_threshold)

        # Encode keyframes with CLIP
        inputs = self.clip_processor(images=keyframes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**inputs).cpu().numpy()

        # Normalize embeddings
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        image_embeddings = image_embeddings.tolist()

        # Query ChromaDB with image embeddings
        image_results = self.collection.query(
            query_embeddings=image_embeddings,
            n_results=n_documents
        )

        # Flatten and deduplicate results
        ids, docs, dists = flatten_query_results(image_results)
        result_dict = {}
        for id_, doc, dist in zip(ids, docs, dists):
            if id_ not in result_dict:
                result_dict[id_] = (doc, dist)

        # Sort by distance and get top-k
        sorted_results = sorted(result_dict.items(), key=lambda item: item[1][1])
        top_k_docs = [doc for _, (doc, _) in sorted_results[:n_documents]]
        return top_k_docs


class ImageRAGRetriever:
    def __init__(
            self,
            chroma_db_path: str = CHROMA_DB_PATH,
            collection_name: str = CHROMA_COLLECTION_IMAGE,
            clip_model_name: str = CLIP_MODEL_NAME
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, device_map="auto")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(
            self,
            question: str,
            image_path: str,
            n_documents: int = 3
    ) -> List[str]:
        """
        Retrieve relevant documents from ChromaDB based on image content

        Args:
            question: User question
            image_path: Path to image file
            n_documents: Number of documents to retrieve

        Returns:
            List of retrieved document strings
        """
        logger.info(f"Retrieving documents for image: {image_path}")

        from PIL import Image

        # Load and encode image
        image = [Image.open(image_path).convert("RGB")]
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**inputs).cpu().numpy()

        # Normalize embeddings
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        image_embeddings = image_embeddings.tolist()

        # Query ChromaDB
        image_results = self.collection.query(
            query_embeddings=image_embeddings,
            n_results=n_documents
        )

        # Flatten results
        ids, docs, dists = flatten_query_results(image_results)
        return docs