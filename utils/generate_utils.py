from typing import List
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


def get_key_frames(video_path: str, hist_threshold: float = 0.8) -> List[np.ndarray]:
    """Extract key frames from video based on histogram difference"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    keyframes = []
    ret, first_frame = cap.read()

    if ret:
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_hist = cv2.normalize(cv2.calcHist([gray], [0], None, [256], [0, 256]), None).flatten()
        keyframes.append(first_frame)
    else:
        cap.release()
        raise ValueError("Failed to read the first frame.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_hist = cv2.normalize(cv2.calcHist([gray], [0], None, [256], [0, 256]), None).flatten()
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

        if diff < hist_threshold:
            keyframes.append(frame)
        prev_hist = curr_hist

    cap.release()
    return keyframes


def compute_max_pooling(
        text: str,
        video_path: str,
        clip_model,
        clip_processor,
        hist_threshold: float = 0.8
) -> float:
    """Compute maximum similarity between text and video keyframes using CLIP"""
    logger.info(f'Max pooling subject: {text}')

    keyframes = get_key_frames(video_path, hist_threshold)
    inputs = clip_processor(images=keyframes, return_tensors="pt").to('cuda')

    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**inputs)
    image_embeddings = image_embeddings.cpu().numpy()
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    text_inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to('cuda')

    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**text_inputs).cpu().numpy()
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_embedding = text_embeddings[0].reshape(1, -1)

    similarities = cosine_similarity(text_embedding, image_embeddings)
    return similarities.max()


def flatten_query_results(results):
    """Flatten ChromaDB query results"""
    if not results:
        return [], [], []
    ids = [i for sublist in results.get("ids", []) for i in sublist]
    docs = [d.replace('\n', '').replace('. ', '') for sublist in results.get("documents", []) for d in sublist]
    distances = [dist for sublist in results.get("distances", []) for dist in sublist]
    return ids, docs, distances


def parse_draft_response(response: str) -> tuple:
    """Parse draft response into answer and reasoning"""
    parts = response.split("Reasoning:")
    if len(parts) >= 2:
        answer = parts[1].strip()
        reasoning = parts[0].strip()
    else:
        answer = response
        reasoning = ""
    return answer, reasoning


def parse_subject_reasoning(answer: str) -> tuple:
    """Parse answer to extract subject and answer text"""
    subject = ""
    reasoning = ""

    if 'Subject:' in answer and 'Reasoning:' in answer:
        subject = answer.split('Reasoning:')[0].split('Subject:')[1].strip()
        reasoning = answer.split('Reasoning:')[1].strip()

    return subject, reasoning


def select_best_draft(rel_scores: List[float], entity_scores: List[float], alpha: float = 0.3) -> int:
    """Select best draft based on reliability and entity scores"""
    max_rel = max(rel_scores)
    candidates = [i for i, r in enumerate(rel_scores) if r >= max_rel - alpha]

    max_ett = max(entity_scores[i] for i in candidates)
    ett_candidates = [i for i in candidates if entity_scores[i] == max_ett]

    if len(ett_candidates) > 1:
        best = max(ett_candidates, key=lambda i: rel_scores[i])
    else:
        best = ett_candidates[0]

    return best