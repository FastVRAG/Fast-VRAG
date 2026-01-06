import logging
from typing import List

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from model.models import Document, RAGDrafter, RAGVerifier
from retrival.retrieval import VideoRAGRetriever
from utils.generate_utils import select_best_draft
from config.generation_config import *

logger = logging.getLogger(__name__)


class SpeculativeRAG:
    def __init__(
            self,
            drafter_model,
            drafter_processor,
            cluster_num: int = CLUSTER_NUM
    ):
        self.drafter = RAGDrafter(drafter_model, drafter_processor)
        self.verifier = RAGVerifier()
        self.cluster_num = cluster_num

    def __call__(self, question: str, documents: List[Document], video_path: str) -> str:
        drafts = self.drafter.generate_draft(question, documents, video_path)
        scores = self.verifier.compute_score(question, drafts, video_path)
        rel_score = scores[0]
        ett_score = scores[1]
        best_idx = select_best_draft(rel_score, ett_score, ALPHA)
        return drafts[best_idx][0]


def initialize_models():
    draft_model = Qwen3VLForConditionalGeneration.from_pretrained(
        DEFAULT_DRAFTER_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    draft_processor = AutoProcessor.from_pretrained(DEFAULT_DRAFTER_MODEL_NAME)
    draft_processor.tokenizer.padding_side = "left"

    rag_pipeline = SpeculativeRAG(draft_model, draft_processor)

    return rag_pipeline


def process_video_question(
        rag_pipeline: SpeculativeRAG,
        question: str,
        video_path: str,
        retriever: VideoRAGRetriever = None,
        n_documents: int = DEFAULT_N_DOCUMENTS
) -> str:
    """
    Process a video question using the RAG pipeline
    """
    # Retrieve documents
    if retriever is None:
        retriever = VideoRAGRetriever()
    retrieved_docs = retriever.retrieve(video_path, n_documents)
    documents = [Document(content=doc) for doc in retrieved_docs]

    # Generate answer
    final_answer = rag_pipeline(question, documents, video_path)

    return final_answer


if __name__ == '__main__':
    # Initialize models
    rag_pipeline = initialize_models()
    retriever = VideoRAGRetriever()

    # Example usage
    question = "What kind of animal is in the video?"
    video_path = "/path/to/video.mp4"

    answer, time_taken = process_video_question(
        rag_pipeline,
        question,
        video_path,
        retriever,
        n_documents=10
    )

    print(f"\nAnswer: {answer}")
    print(f"Time: {time_taken:.2f}s")