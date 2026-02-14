# 🇮🇳 Llama-3 Indian Gender Classifier

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/shisha-07/Llama-3-Indian-Gender-Classifier)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/shisha-07/Indian_Names_with_Gender_Dataset)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHIVASHANKAR-V07/Llama_3_Indian_Gender_Classifier/blob/main/notebooks/Gender_Classifier_Training.ipynb)
## 📌 Project Overview
This repository contains the training code and resources for the **Indian Name Gender Classifier**, a fine-tuned [🦙 Llama-3 model](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit) capable of classifying names as **Male**, **Female**, or **Neutral** with high precision.

The model was trained using [🦥 Unsloth](https://github.com/unslothai/unsloth) on a balanced dataset of 42,000 Indian names.

## 🔗 Quick Links
* **🤖 Model Weights:** [Hugging Face Model Hub](https://huggingface.co/shisha-07/Llama-3-Indian-Gender-Classifier)
    * Available in **GGUF**, **Merged (16-bit)**, and **LoRA** formats.
* **📚 Training Dataset:** [Hugging Face Datasets](https://huggingface.co/datasets/shisha-07/Indian_Names_with_Gender_Dataset)
    * Contains 42k names labeled (0=NeutralOther, 1=Male, 2=Female).

## 🛠️ Usage
To run the model locally or in Colab, check the notebook provided above.

### Installation
```bash
pip install unsloth torch transformers
```

### Inference Code
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "shisha-07/Llama-3-Indian-Gender-Classifier",
    subfolder = "LoRA", 
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# ... (See notebook for full code)
```

## **📜 License**
Apache-2.0

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
