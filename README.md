# Python Char-Level Code Autocomplete LLM Model

A lightweight **character-level Python code autocompletion** system built with a **GRU-based Recurrent Neural Network**, trained on the **CoNaLa** dataset.  
Given a Python code prompt, the model performs **autoregressive next-character prediction** to generate realistic code completions.  
Optimized for **fast inference** and deployed with a **FastAPI web interface**.

---

## Features
- **Character-level language model** for Python code.
- **GRU-based RNN** for sequence modeling.
- Trained on **CoNaLa (Code/Natural Language Challenge)** dataset.
- **Autoregressive** generation — predicts next characters one by one.
- **FastAPI web app** with Jinja2 templates for easy interaction.
- Easy to train, save, and load the model.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/HitendraSinghPanwar/python-char-code-autocomplete-llm-model.git
cd python-char-code-autocomplete-llm-model
```

### Create & Activate Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset
- Default: **CoNaLa dataset** (JSONL format).
- You can replace `FILE_PATH` in `config.py` with your own dataset.
- If no dataset is found, dummy Python snippets will be used.

---

## Training the Model
Run:
```bash
python train.py
```
This will:
- Load dataset from `config.FILE_PATH`
- Preprocess text into sequences
- Train the GRU-based char-RNN
- Save:
  - Model weights → `char_autocomplete_model.pth`
  - Character mappings → `char_mappings.json`

---

## Running Inference (Web App)
Once trained:
```bash
python app.py
```
- Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Enter a Python code prompt, e.g.:
  ```
  def factorial(n):
  ```
- Click **Autocomplete** to get predicted code continuation.

---

## Configuration
Modify hyperparameters in **`config.py`**:
```python
SEQ_LENGTH = 100       # Input sequence length
HIDDEN_SIZE = 256      # GRU hidden units
NUM_LAYERS = 2         # GRU layers
LEARNING_RATE = 0.002  # Optimizer learning rate
NUM_EPOCHS = 10        # Training epochs
BATCH_SIZE = 128       # Training batch size
```

---

## Inspiration
- Inspired by **char-RNN** by Andrej Karpathy.
- Dataset: [CoNaLa Challenge](https://conala-corpus.github.io/)

---

## Author
**Hitendra Singh Panwar**  
[GitHub Profile](https://github.com/HitendraSinghPanwar)
