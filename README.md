<div align="center">

<h1>🤖 Hands-on AI Tutorial</h1>

<p><strong>Learn it. Build it. Understand the math behind it.</strong></p>

<p><em>A fully open-source, beginner-friendly AI course — from Linear Regression to Deep Learning, built from scratch.</em></p>

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-orange?style=flat-square)]()
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=flat-square)]()

<br/>

| 📚 Chapters | 🧠 Tracks | 🔢 Algorithms | 🎯 Audience |
|:-----------:|:---------:|:-------------:|:-----------:|
| **5+** | **2** | **10+** | Beginners → Intermediates |

<br/>

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-189AB4?style=flat-square)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

<br/>

**[🚀 Get Started](#-getting-started)** · **[📦 What's Inside](#-whats-inside)** · **[🗂️ Structure](#️-project-structure)** · **[🧭 Learning Path](#-recommended-learning-path)** · **[🤝 Contribute](#-contributing)**

<br/>

---

> *"You don't truly understand an algorithm until you've implemented it yourself."*

</div>

---

## 🎯 Who Is This For?

| Level | What You Bring | What You'll Get |
|---|---|---|
| 🟢 **Beginner** | Basic Python knowledge | Your first working ML models + the math explained |
| 🟡 **Junior** | Some ML exposure | Deep understanding of *how* and *why* algorithms work |

No prior AI or ML experience needed. Basic Python and high-school-level math is all you need to get started.

---

## 📦 What's Inside?

This repository is split into **two learning tracks**:

### 🔬 Track 1: Machine Learning Fundamentals (`ml_fundamentals/`)

Classical machine learning — the building blocks of all AI.

| Chapter | Topic | What You'll Build |
|---|---|---|
| **Chapter 1** | [📈 Regression](ml_fundamentals/chapter1/README.md) | Predict continuous values (house prices, life expectancy) |
| **Chapter 2** | 🏷️ Classification *(coming soon)* | Categorize data (spam detection, disease diagnosis) |
| **Chapter 3** | 🔵 Clustering *(coming soon)* | Group unlabeled data (customer segmentation) |

### 🧠 Track 2: Deep Learning (`deep_learning/`)

Neural networks and modern AI architectures.

| Chapter | Topic | What You'll Build |
|---|---|---|
| **Chapter 4** | 💬 NLP *(coming soon)* | Make machines understand text and language |
| **Chapter 5** | 👁️ Computer Vision *(coming soon)* | Make machines see and interpret images |

---

## 🧠 What You Will Learn

- **📊 Data Handling** — Load, clean, and preprocess real-world datasets
- **🔢 The Math** — Intuitive breakdowns of the equations behind each algorithm
- **🛠️ From-Scratch Implementations** — Build models with NumPy to understand *how* they work
- **🤝 Library Usage** — Apply scikit-learn and industry tools for real-world use
- **📈 Evaluation** — Measure and compare model performance with the right metrics
- **💾 Model Persistence** — Save, load, and reuse trained models

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Hands-on-AI-Tutorial.git
cd Hands-on-AI-Tutorial
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost joblib
```

### 3. Start Learning

Open the first chapter and follow the guide:

```
ml_fundamentals/chapter1/
```

Or launch the interactive notebook directly:

```bash
jupyter notebook ml_fundamentals/chapter1/notebooks/Regression.ipynb
```

---

## 🗂️ Project Structure

```
Hands-on-AI-Tutorial/
│
├── README.md                       ← You are here
│
├── ml_fundamentals/                ← Classical ML track
│   ├── chapter1/                   ← Regression (✅ Active)
│   │   ├── configs/                ← Model & dataset config (JSON)
│   │   ├── data/                   ← Datasets (CSV files)
│   │   ├── docs/                   ← Deep-dive algorithm docs with math
│   │   ├── models/                 ← Saved trained model files
│   │   ├── notebooks/              ← Interactive Jupyter notebooks
│   │   ├── results/                ← Reports and plots generated after training
│   │   ├── src/                    ← From-scratch Python implementations
│   │   └── README.md               ← Chapter guide
│   ├── chapter2/                   ← Classification (🔜 Coming soon)
│   └── chapter3/                   ← Clustering (🔜 Coming soon)
│
└── deep_learning/                  ← Deep Learning track
    ├── chapter4/                   ← NLP (🔜 Coming soon)
    └── chapter5/                   ← Computer Vision (🔜 Coming soon)
```

---

## 🧭 Recommended Learning Path

```
1. Chapter 1 — Regression
   ├── Read: ml_fundamentals/chapter1/README.md
   ├── Explore: notebooks/Regression.ipynb
   ├── Go deeper: docs/ (math + implementation guides)
   └── Run: src/ scripts directly

2. Chapter 2 — Classification (coming soon)
3. Chapter 3 — Clustering (coming soon)
4. Chapter 4 — NLP (coming soon)
5. Chapter 5 — Computer Vision (coming soon)
```

---

## 🤝 Contributing

This is a fully open-source community project. Contributions are warmly welcome!

- 💡 Found a bug or unclear explanation? **Open an issue.**
- 📖 Want to add a chapter or improve a doc? **Submit a pull request.**
- ⭐ If this helps you, **give it a star** — it helps others find it too!

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ for learners everywhere · Open Source · MIT License</sub>
</div>
