# 💳 Fraud Detection Project

This project builds a machine learning pipeline to detect fraudulent credit card transactions, using a real-world dataset.
---

## 🚀 Installation Guide

### 1️⃣ Navigate to the project folder
```bash
cd ./fraud_detection/
```
### 2️⃣ Create a virtual environment
```bash
python -m venv venv
```
### 3️⃣ Activate the environment
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

### 4️⃣ Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 5️⃣ Install required packages (first-time users)

- Install PyTorch with CUDA 12.6 (nightly build):
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

- Install project dependencies:
```bash
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn transformers jupyterlab seaborn matplotlib pytest pytest-dash black flake8
```

### 6️⃣ Save installed packages (optional, recommended)
```bash
pip freeze > requirements.txt
```

### 7️⃣ To reproduce the environment later
```bash
pip install -r requirements.txt
```

### 8️⃣ (Optional) Register as Jupyter kernel
```bash
python -m ipykernel install --user --name=fraud-detection
```

⚠ **Note:**  
The `fraud_detection/data/` folder is excluded from the repository.  
Please download the required datasets manually and place them in this folder.

---

## 📊 Dataset

The dataset used comes from the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.