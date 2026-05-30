# Image Captioning Refinement System Using Deep Learning and Computer-Vision

## Project Overview
A multi-stage image caption refinement pipeline applied to the 
Flickr8k dataset using BLIP embeddings, multi-signal scoring, 
majority voting, and Qwen2.5 caption fusion.

## Live Demo
[HuggingFace Spaces](https://huggingface.co/spaces/Afsha001/Image_captioning)

## Pipeline
1. BLIP — Image & caption embedding extraction (8091, 256)
2. BLIP ITM — Image-text matching scoring
3. Jina Reranker M0 — Cross-encoder relevance scoring
4. Cosine Similarity — Embedding space alignment
5. Majority Voting — Consensus top-2 selection
6. Qwen2.5-1.5B-Instruct — Caption fusion

## Dataset
- Flickr8k — 8,092 images, 5 captions each, 40,460 total annotations

## Evaluation Results
| Method | BLEU-1 | METEOR | CIDEr |
|--------|--------|--------|-------|
| Random Caption | 1.000 | 0.999 | 2.641 |
| Best Cosine | 0.977 | 0.922 | 2.498 |
| Qwen Fused | 0.648 | 0.697 | 0.697 |

## Tech Stack
Python · PyTorch · HuggingFace Transformers · Streamlit · BLIP · Qwen2.5 · Jina AI

## University
M.Sc. Data Science — Aligarh Muslim University, 2025–26
