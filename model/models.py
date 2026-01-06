import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    CLIPProcessor,
    CLIPModel
)

from config.generation_config import *
from utils.generate_utils import compute_max_pooling, parse_draft_response, parse_subject_reasoning

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    source: str = ""
    score: float = 0.0


class RAGDrafter:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def generate_draft(
            self,
            question: str,
            documents: List[Document],
            video_path: str
    ) -> List[Tuple[str, str]]:
        drafts = []
        messages = []

        for doc in documents:
            evidence = doc.content
            prompt = GENERATION_PROMPT_TEMPLATE.format(question=question, evidence=evidence)

            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages.append(message)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            fps=VIDEO_FPS
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=200)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i, response in enumerate(output_texts):
            answer, reasoning = parse_draft_response(response)
            drafts.append((answer, reasoning))
        return drafts


class RAGVerifier:
    def __init__(self, model_name: str = DEFAULT_VERIFIER_MODEL_NAME):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"

        self.clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME,
        ).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME,
        )


    def compute_score(
            self,
            question: str,
            drafts: List[Tuple[str, str]],
            video_path: str
    ) -> List[List[float]]:
        messages = []

        for (answer, reasoning) in drafts:
            subject, reasoning = parse_subject_reasoning(reasoning)
            prompt = VERIFY_PROMPT_TEMPLATE.format(
                question=question,
                subject=subject,
                reasoning=reasoning,
                answer=answer
            )

            messages_forward = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": VIDEO_MAX_PIXELS,
                            "fps": VIDEO_FPS,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages.append(messages_forward)

        inputs_forward = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            fps=VIDEO_FPS
        )
        inputs_forward = inputs_forward.to("cuda")

        yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]

        with torch.no_grad():
            outputs = self.model(**inputs_forward)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits.float(), dim=-1)
            yes_prob = probs[:, yes_token_id]
            no_probs = probs[:, no_token_id]

            total = yes_prob + no_probs
            yes_prob_norm = yes_prob / total
            reliable_score = yes_prob_norm.tolist()

        entity_score = []
        for answer, reasoning in drafts:
            subject, _ = parse_subject_reasoning(reasoning)
            if subject:
                score = compute_max_pooling(
                    subject,
                    video_path,
                    self.clip_model,
                    self.clip_processor,
                    KEYFRAME_HIST_THRESHOLD
                )
                entity_score.append(score)
            else:
                entity_score.append(0.0)

        return [reliable_score, entity_score]