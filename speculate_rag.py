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
    import argparse
    parser = argparse.ArgumentParser(description='Speculative RAG for Video QA')
    parser.add_argument(
        '--question',
        type=str,
        required=True,
        help='Question about the video'
    )
    parser.add_argument(
        '--video-path',
        type=str,
        required=True,
        help='Path to video file'
    )
    parser.add_argument(
        '--n-documents',
        type=int,
        default=DEFAULT_N_DOCUMENTS,
        help=f'Number of documents to retrieve (default: {DEFAULT_N_DOCUMENTS})'
    )
    parser.add_argument(
        '--frame-threshold',
        type=float,
        default=DEFAULT_FRAME_THRESHOLD,
        help=f'Threshold for keyframe extraction (default: {DEFAULT_FRAME_THRESHOLD})'
    )
    parser.add_argument(
        '--chroma-db-path',
        type=str,
        default=CHROMA_DB_PATH,
        help=f'Path to ChromaDB (default: {CHROMA_DB_PATH})'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default=CHROMA_COLLECTION_NAME,
        help=f'ChromaDB collection name (default: {CHROMA_COLLECTION_NAME})'
    )
    args = parser.parse_args()

    # Initialize models
    rag_pipeline = initialize_models()
    retriever = VideoRAGRetriever(
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name
    )

    # Process question
    answer, cost = process_video_question(
        rag_pipeline,
        args.question,
        args.video_path,
        retriever,
        n_documents=args.n_documents
    )

    print(f"Answer: {answer}")
    print(f"Time: {cost:.2f}s")