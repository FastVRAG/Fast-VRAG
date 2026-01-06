DEFAULT_VERIFIER_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DRAFTER_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
CLIP_MODEL_NAME = 'openai/clip-vit-large-patch14-336'

# ChromaDB
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "knowledge_title_video"
CHROMA_COLLECTION_IMAGE = "knowledge_title_image"

# Video processing
VIDEO_MAX_PIXELS = 360 * 480
VIDEO_FPS = 1.0
KEYFRAME_HIST_THRESHOLD = 0.8
DEFAULT_FRAME_THRESHOLD = 0.6

# RAG parameters
NUM_DRAFTS = 5
CLUSTER_NUM = 3
ALPHA = 0.3
DEFAULT_N_DOCUMENTS = 10

# Prompts
GENERATION_PROMPT_TEMPLATE = """You are an intelligent assistant. First, identify the main subject in the video, then provide reasoning based on the subject and the evidence, and finally give the answer. 
## Output Format: 
Subject:
Reasoning:
Answer:

## Instruction: 
Question: {question}
Evidence: {evidence}
"""

VERIFY_PROMPT_TEMPLATE = """You are a verification assistant. Your task is to determine whether the answer is reasonable and internally consistent.
## Output Format:
Yes/No

## Instruction:  
Question: {question} 
Subject: {subject}
Reasoning: {reasoning} 
Answer: {answer}
"""