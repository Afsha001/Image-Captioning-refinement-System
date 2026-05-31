[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Afsha001/Image_captioning)
<div align="center">

# 🖼️ Image Captioning Refinement Using Deep Learning

### A multi-stage pipeline that generates, scores, votes and fuses image captions into one refined, grounded description

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/Afsha001/Image_captioning)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
![REST API](https://img.shields.io/badge/REST_API-Jina-009191?style=flat&logo=fastapi&logoColor=white)
**M.Sc. Data Science — Aligarh Muslim University, Aligarh · 2025–26**

</div>

---

## 📌 Project Overview

Standard image captioning models produce a single output with no quality check — often generic, incomplete, or hallucinated. This project proposes an **Image Caption Refinement System** that treats captioning as a multi-stage quality improvement process:

- Generates **5 diverse candidate captions** from an input image
- Evaluates each candidate through **3 independent scoring signals**
- Selects the **top-2 by majority voting consensus**
- Fuses them into **one refined, comprehensive caption** using a language model

The system was built and evaluated on the **Flickr8k dataset** (8,091 images · 40,455 captions) and deployed as a live web application on HuggingFace Spaces.

---

## 🚀 Live Demo

> Upload any image and get a refined, detailed caption in seconds.

👉 **[Try the Live App on HuggingFace Spaces](https://huggingface.co/spaces/Afsha001/Image_captioning)**

---

## 🗂️ Dataset
<img width="311" height="320" alt="image" src="https://github.com/user-attachments/assets/a9af6ca1-7bb3-44a6-9923-40d6e4c45fee" />

---

## 🧠 Models Used

| Model | Role | Type |
|---|---|---|
| Florence-2-Large | Caption generation | Local |
| BLIP (Salesforce) | Embedding extraction + ITM scoring | Local |
| Jina Reranker M0 | Cross-encoder relevance scoring | API |
| Qwen2.5-1.5B-Instruct | Caption fusion | Local |

---

## 📊 Results & Evaluation

### Evaluation Metrics — BLEU · METEOR · CIDEr

Refined captions evaluated against 5 human reference captions per image.

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|---|---|---|
| Random Caption | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 2.641 |
| Best Cosine | 0.977 | 0.971 | 0.964 | 0.957 | 0.922 | 2.498 |
| **Qwen Fused** | **0.648** | **0.536** | **0.440** | **0.353** | **0.697** | **0.697** |

> **Note:** Random Caption scores 1.0 because it IS one of the 5 references — a trivial baseline.
> Qwen Fused generates novel, independent descriptions — more complete and descriptive.

---

### Cosine Similarity Score Distribution

BLIP embedding cosine similarity computed between each image and its 5 captions across all 8,091 images.

| Score Type | Mean | Std |
|---|---|---|
| Best Caption | 0.5360 | 0.0412 |
| Worst Caption | 0.4452 | 0.0389 |
| Average | 0.4912 | — |


---

### Majority Voting Results

All 8,091 images processed · **0 skipped** · Strong inter-signal consensus confirmed.

<img width="550" height="525" alt="majority_voting" src="https://github.com/user-attachments/assets/3ef4503c-2bd0-4fe3-b9ca-1a383b4c9321" />


---

### Qwen Caption Fusion — Sample Outputs

<img width="550" height="525" alt="Qwen_fusion" src="https://github.com/user-attachments/assets/ab89acfd-c0ab-474d-8d54-6a65eef3a3c3" />



> **Example:**
> - Cap 1: *a little girl in a pink dress going into a wooden cabin*
> - Cap 2: *a little girl climbing the stairs to her playhouse*
> - **Fused:** *A little girl in a pink dress climbs into a wooden playhouse near a tree, entering through its door.*

---

### BLIP Embedding Extraction

| Embedding | Shape | Notes |
|---|---|---|
| Image Embeddings | (8091, 256) | L2-normalised |
| Caption Embeddings | (40455, 256) | L2-normalised |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch · HuggingFace Transformers |
| Vision Models | Florence-2-Large · BLIP |
| Language Models | Qwen2.5-1.5B-Instruct |
| Reranking | Jina Reranker M0 |
| Scoring | scikit-learn · Cosine Similarity |
| Evaluation | NLTK · BLEU · METEOR · pycocoevalcap · CIDEr |
| Web App | Streamlit |
| Deployment | HuggingFace Spaces |
| Development | Google Colab · Google Drive |

---

## 📁 Repository Structure
## 📂 Project Structure
```
image-captioning-refinement/
├── app.py                          # HuggingFace Streamlit application
├── requirements.txt                # Python dependencies
├── Image_captioning_notebook.ipynb # Full research pipeline notebook
├── README.md                       # Documentation
├── LICENSE                         # Open-source license
└── results/                        # Experimental charts & visualizations
    ├── cosine_distribution.png     # Visualizing BLIP embedding similarity
    ├── baseline_comparison.png     # Comparison against standard benchmarks
    ├── majority_voting.png         # Consensus stage performance
    ├── Qwen_fusion.png             # LLM refinement output analysis
    └── combined_scores.png         # BLEU, CIDEr, and METEOR metrics
```
---

## ▶️ Run Locally

```bash
git clone https://github.com/Afsha001/image-captioning-refinement
cd image-captioning-refinement
pip install -r requirements.txt
streamlit run app.py
```

> Add your `JINA_KEY` as an environment variable before running.

---

## 🏛️ Academic Details

| Field | Details |
|---|---|
| Degree | M.Sc. Data Science |
| University | Aligarh Muslim University, Aligarh |
| Department | Statistics & Operations Research |
| Internship Centre | Interdisciplinary Centre for Artificial Intelligence |
| Supervisor | Dr. Junaid Ali Reshi |
| Co-Supervisors | Dr. Ahmad Yusuf Adhami · Dr. Mohd. Faizan · Dr. Shazia Farhin |
| Session | 2025–26 |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **Afsha Anjum** · AMU Aligarh · 2025–26

</div>    
     

     
