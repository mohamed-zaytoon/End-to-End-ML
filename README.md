# 🧠 End-to-End ML: News Article Tagging System

This project implements a full end-to-end machine learning pipeline that scrapes news articles, tags them using a Question-Answering (QA) format, trains a fine-tuned model using LoRA, tracks experiments via MLflow, evaluates the system, and finally serves the model using vLLM.

---

##  Project Overview

- **Objective:** Automatically tag news articles (e.g., *sports*, *art*, *world*) using a QA-based approach.
- **Methodology:**
  - Scrape articles from public news sources.
  - Generate QA pairs where:
    - **Question**: The article body.
    - **Answer**: The list of relevant tags.
  - Split the dataset into train, validation, and test sets.
  - Fine-tune the `Qwen2.5 0.5B` model using **LoRA** for efficiency.
  - Use **MLflow** for experiment tracking.
  - Evaluate model predictions using **BLEU** and **ROUGE** scores.
  - Serve the final model with **vLLM** for scalable inference.

---

##  Tech Stack

- **Scraping**: `requests`, `BeautifulSoup`
- **QA Pairing**: Custom preprocessing logic
- **Model**: `Qwen2.5-0.5B`
- **Fine-Tuning**: `LoRA` (via `PEFT`)
- **Experiment Tracking**: `MLflow`
- **Evaluation**: `BLEU`, `ROUGE`
- **Model Serving**: `vLLM`

---

##  Data Pipeline

```mermaid
graph LR
A[Scrape Articles] --> B[Generate QA Pairs]
B --> C[Split Dataset (Train/Val/Test)]
```

## 🔧 Training Setup

- **Model**: Qwen2.5 (0.5B)
- **Fine-tuning**: LoRA
- **Tracking**: MLflow for logging metrics, configs, and artifacts

---

##  Evaluation Metrics

- **BLEU Score**: Measures precision of predicted tags.
- **ROUGE Score**: Measures recall and overlap with ground-truth tags.

---

##  Serving

- **Serving Engine**: vLLM
