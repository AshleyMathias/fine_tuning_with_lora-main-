# 🤖 Train BERT with LoRA — Streamlit App

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-BERT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co)
[![LoRA](https://img.shields.io/badge/LoRA-Adapter%20Tuning-blueviolet)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/github/license/AshleyMathias/fine_tuning_with_lora-main-)](LICENSE)

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

Here’s a preview of the Streamlit interface where users can input sentences, label sentiment, and train a BERT model with LoRA adapters — all in the browser!

![App Screenshot](screenshots/Screenshot.png)

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
git clone https://github.com/AshleyMathias/fine_tuning_with_lora-main-.git
cd fine_tuning_with_lora-main-
```

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

---

## 📂 Project Structure

bert-lora-streamlit/
├── app.py # Streamlit app interface
├── fine_tuning2.py # Core training and inference logic
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── LICENSE # MIT License file
└── screenshots/
└── app_preview.png # UI screenshot for README

---

## ✨ Sample Use Case

> Imagine building an educational tool for workshops or GenAI learning labs where students can write a few sentences, label them, and watch a model learn to classify new inputs. This is exactly that — a **real-time, interactive ML demo** powered by LoRA and BERT.

💡 Use it for:
- Interactive AI training experiences
- LoRA adapter learning demos
- Quick prototyping for NLP pipelines
- Lightweight fine-tuning without cloud GPU

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

If you’d like to:
- Suggest improvements
- Report bugs
- Add new features or training options

Feel free to fork the repository and submit a pull request.  
Please follow standard commit practices and open issues with context.

**Let’s make it better together! 🚀**

---

## 📄 License

This project is licensed under the **MIT License**.

You’re free to use, modify, and distribute this codebase — just retain attribution.  
See the [LICENSE](LICENSE) file for details.
---

<div align="center">

📘 _Built for learning, prototyping, and showcasing what fine-tuning with adapters can do — even without a GPU._

<br/>

🔗 **Connect with me**  
<a href="https://www.linkedin.com/in/ashleymathias10" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-Ashley%20Mathias-blue?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
&nbsp;&nbsp;&nbsp;
<a href="mailto:ashleymathias100@gmail.com"><img src="https://img.shields.io/badge/Email-Contact%20Me-ff69b4?style=flat&logo=gmail&logoColor=white" alt="Email"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://github.com/AshleyMathias"><img src="https://img.shields.io/badge/GitHub-@AshleyMathias-181717?style=flat&logo=github&logoColor=white" alt="GitHub"></a>

<br/><br/>

🚀 _Want to collaborate, learn together, or use this for workshops?_  
Let’s connect and make it meaningful.

</div>

