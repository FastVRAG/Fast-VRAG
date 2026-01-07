# 🚀FastV-RAG: Towards Fast and Fine-Grained Video QA with Retrieval-Augmented Generation
## Introduction
FastV-RAG is a speculative decoding–based RAG framework for vision–language models (VLMs), targeting efficient and reliable knowledge-intensive video question answering.

The core idea is to decouple drafting and verification: a lightweight model handles multimodal retrieval and candidate answer generation, while a heavyweight model is invoked only to verify and calibrate the results. Applied to VLM–RAG systems such as KVQA, this design enables fast answer proposal from retrieved documents and accurate refinement with strong visual grounding, significantly reducing inference cost compared to standard sequential decoding.

To mitigate errors caused by fine-grained entity confusion in retrieved knowledge, VideoSpeculateRAG introduces a lightweight entity alignment mechanism. The draft model explicitly extracts entities and reasoning traces, and the verifier evaluates candidate answers by measuring CLIP-based similarity between entities and video frames, ensuring both factual correctness and visual consistency.

Experiments on VideoSimpleQA and Encyclopedic VQA demonstrate that VideoSpeculateRAG achieves comparable or improved accuracy over standard RAG approaches while providing nearly a 2× inference speedup, highlighting its effectiveness for efficient multimodal reasoning.
![overview](/assets/overview.png)
## Demo

https://github.com/user-attachments/assets/82da808a-f47d-47b1-8d31-970ce15d40ff

Here is a demo of our system. The left side of the screen shows the generation process of our speculative FastV-RAG model, and the right side shows the original Qwen2.5-VL model.

## Quick Start
+ Build a knowledge base
```bash
python build_index.py \
    --wiki-json /path/to/wiki.json \
    --chroma-db-path /path/to/db \
    --batch-size 128 
```
+ Run the speculative decoding
```bash
python speculative_rag.py \
    --question "Your question" \
    --video-path "path/to/video"
```

