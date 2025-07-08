# 🤖 Train BERT with LoRA — Streamlit App

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-BERT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co)
[![LoRA](https://img.shields.io/badge/LoRA-Adapter%20Tuning-blueviolet)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/github/license/yourusername/bert-lora-streamlit)](LICENSE)

A lightweight web application that allows users to **fine-tune a pre-trained BERT model using LoRA adapters** directly from a Streamlit UI — with just a few training sentences! Ideal for learning, experimentation, and demos without GPU dependency.

---

## 🚀 Features

- Train `bert-base-uncased` on your own sentiment data
- Utilizes **LoRA (Low-Rank Adaptation)** for efficient fine-tuning
- Fully CPU-compatible (no GPU needed)
- Streamlit-powered browser interface
- Predict sentiment from new text instantly after training
- Show class probabilities and model confidence
- Includes example training data injection for quick testing

---

## 🖼️ Screenshot

| Train LoRA BERT on Your Own Sentences |
|---------------------------------------|
| ![App Screenshot](screenshots/app_preview.png) |

---

## 🧱 Tech Stack

- `Transformers` (Hugging Face)
- `PEFT` (Parameter-Efficient Fine-Tuning)
- `Streamlit`
- `PyTorch`
- `Datasets`

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/bert-lora-streamlit.git
cd bert-lora-streamlit
```

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
