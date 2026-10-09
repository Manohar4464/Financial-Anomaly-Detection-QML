# 🚀 Financial Anomaly Detection using Quantum Machine Learning (QML)

> **AQVH 2025 Semi-Finalist Project**  
> A hybrid Quantum Machine Learning (QML) approach for detecting fraudulent financial transactions using **Quantum Support Vector Classifier (QSVC)** and **Fidelity Quantum Kernel**, with an interactive dashboard for real-time anomaly prediction.

---

## 📖 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Why Quantum Machine Learning?](#why-quantum-machine-learning)
- [Key Features](#key-features)
- [Project Workflow](#project-workflow)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Dataset Description](#dataset-description)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Quantum Computing Components](#quantum-computing-components)
- [Performance Evaluation](#performance-evaluation)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

---

# Project Overview

Financial fraud has become one of the fastest-growing cybercrimes worldwide. Millions of digital transactions occur every second, making it increasingly difficult for traditional systems to accurately distinguish between legitimate and fraudulent activities.

This project presents a **Quantum Machine Learning (QML)** based financial anomaly detection system that combines classical preprocessing techniques with quantum-enhanced classification using **Quantum Support Vector Classifier (QSVC)**.

Unlike conventional machine learning models that rely entirely on classical kernels, this project leverages a **Quantum Kernel** to project financial transaction data into a high-dimensional quantum feature space. This enables the model to identify complex decision boundaries that may be difficult for traditional algorithms to capture.

The system also includes an interactive **Dash web dashboard**, allowing users to enter transaction details and instantly receive anomaly predictions.

This project was developed and presented during the **AQVH 2025 Semi-Finals**.

---

# Problem Statement

Financial institutions process millions of transactions daily. Among these transactions, fraudulent activities often resemble legitimate transactions, making detection extremely challenging.

Traditional machine learning models often struggle because:

- Fraud data is highly imbalanced.
- Fraudsters continuously change their attack patterns.
- Nonlinear relationships exist between transaction features.
- Conventional kernels may fail to separate complex transaction patterns.

Our objective is to develop an intelligent fraud detection system capable of identifying suspicious financial transactions more effectively using Quantum Machine Learning.

---

# Objectives

The primary objectives of this project are:

- Detect fraudulent financial transactions.
- Improve anomaly detection accuracy.
- Explore the potential of Quantum Machine Learning.
- Compare Quantum ML with Classical ML.
- Build a real-time prediction dashboard.
- Demonstrate practical applications of quantum computing in cybersecurity.

---

# Why Quantum Machine Learning?

Quantum Machine Learning combines principles from:

- Quantum Computing
- Machine Learning
- Data Science

Instead of processing data only in classical feature space, QML maps data into a **Quantum Hilbert Space**, where complex relationships become easier to separate.

### Advantages

- Better nonlinear feature representation
- High-dimensional quantum feature mapping
- Stronger decision boundaries
- Potential scalability with future quantum hardware
- Research-oriented approach

Although current quantum computers are still evolving, hybrid quantum-classical algorithms such as QSVC have shown promising results for specialized classification tasks.

---

# Key Features

- Quantum Support Vector Classifier (QSVC)
- Fidelity Quantum Kernel
- ZZFeatureMap encoding
- Financial transaction preprocessing
- Automatic feature engineering
- Threshold optimization
- Random Forest baseline comparison
- Confusion Matrix visualization
- ROC-AUC evaluation
- Interactive Dash Dashboard
- Real-time anomaly prediction

---

# Project Workflow

```
Financial Transactions Dataset
                │
                ▼
      Data Preprocessing
                │
                ▼
      Feature Engineering
                │
                ▼
     Numerical Standardization
                │
                ▼
      Categorical Encoding
                │
                ▼
     Quantum Feature Mapping
                │
                ▼
      Fidelity Quantum Kernel
                │
                ▼
 Quantum Support Vector Classifier
                │
                ▼
      Threshold Optimization
                │
                ▼
      Performance Evaluation
                │
                ▼
      Interactive Dashboard
```

---

# System Architecture

```
                  +-----------------------+
                  | Financial Dataset     |
                  +-----------------------+
                             |
                             |
                  Data Cleaning & Sorting
                             |
                             |
                Feature Engineering Module
                             |
                             |
             StandardScaler + OneHotEncoder
                             |
                             |
                Quantum Feature Mapping
                  (ZZFeatureMap)
                             |
                             |
               Fidelity Quantum Kernel
                             |
                             |
                     QSVC Training
                             |
               +-------------+-------------+
               |                           |
               |                           |
      Model Evaluation             Dashboard Prediction
               |                           |
               +-------------+-------------+
                             |
                      User Prediction
```

---

# Technology Stack

## Programming Language

- Python

## Machine Learning

- Scikit-Learn

## Quantum Computing

- Qiskit
- Qiskit Machine Learning
- Fidelity Quantum Kernel
- QSVC

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

## Dashboard

- Plotly Dash

---

# Dataset Description

The dataset contains financial transaction records including:

| Feature | Description |
|----------|-------------|
| amount | Transaction amount |
| type | Type of transaction |
| transaction_time | Time of transaction |
| user_id | Unique customer ID |
| is_anomaly | Target variable |

---

# Feature Engineering

To improve model performance, several additional behavioral features were generated.

### Transaction Amount

Represents the monetary value involved in each transaction.

---

### Hour

Extracted from transaction timestamp.

Helps identify unusual transaction timings.

---

### Day of Week

Captures weekly transaction behavior.

---

### Time Since Last Transaction

Measures the interval between consecutive transactions.

Useful for identifying rapid suspicious activities.

---

### Transaction Frequency

Represents the total number of transactions made by a customer.

Frequent abnormal activities often indicate fraud.

---

# Machine Learning Pipeline

## Step 1

Load Dataset

↓

## Step 2

Preprocess Data

↓

## Step 3

Generate New Features

↓

## Step 4

Normalize Numerical Features

↓

## Step 5

Encode Transaction Type

↓

## Step 6

Split Dataset

- Training
- Validation
- Testing

↓

## Step 7

Train QSVC

↓

## Step 8

Optimize Threshold

↓

## Step 9

Evaluate Performance

↓

## Step 10

Launch Dashboard

---

# Quantum Computing Components

## ZZFeatureMap

Encodes classical financial data into quantum states.

This creates richer feature representations for classification.

---

## Fidelity Quantum Kernel

Measures similarity between quantum states.

Provides a more expressive kernel than many classical approaches.

---

## Quantum Support Vector Classifier (QSVC)

The QSVC uses the quantum kernel to separate normal and fraudulent transactions.

Benefits include:

- Better nonlinear classification
- Quantum feature space
- Research-focused approach
- Hybrid quantum-classical workflow

---

# Performance Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

These metrics provide a comprehensive assessment of fraud detection performance.

---

# Dashboard

The project includes an interactive dashboard built using **Dash**.

Users can:

- Enter transaction amount
- Select transaction type
- Input transaction features
- Predict anomalies instantly

The dashboard displays:

- Prediction result
- Anomaly score
- Transaction classification

---

# Project Structure

```
Financial-Anomaly-Detection-QML/

│
├── README.md
│
├── docs/
│   ├── 01_problem_statement.txt
│   ├── 02_approach_and_why_qml.txt
│   ├── 03_code_explanation.txt
│   ├── 04_contributors.txt
│
├── src/
│   └── anomaly_detection_qsvc.py
│
├── dataset/
│   └── sample_dataset.csv
│
├── images/
│   ├── architecture.png
│   ├── dashboard.png
│   ├── workflow.png
│   └── confusion_matrix.png
│
├── requirements.txt
│
└── LICENSE
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Financial-Anomaly-Detection-QML.git

cd Financial-Anomaly-Detection-QML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the application:

```bash
python anomaly_detection_qsvc.py
```

The Dash dashboard will launch locally.

---

# Results

The project demonstrates that combining classical preprocessing with Quantum Machine Learning can effectively identify anomalous financial transactions.

Key achievements include:

- Hybrid quantum-classical fraud detection pipeline
- Interactive prediction dashboard
- Quantum kernel-based classification
- Comparative evaluation with Random Forest
- Improved understanding of QML applications in cybersecurity

---

# Future Improvements

Potential enhancements include:

- Integration with IBM Quantum hardware
- Larger financial datasets
- Deep Quantum Neural Networks
- Federated fraud detection
- Explainable AI integration
- Blockchain-based transaction verification
- Real-time streaming analytics
- Cloud deployment using Docker and Kubernetes

---

# Contributors

This project was developed as part of the **AQVH 2025 Semi-Finals**.

| Name | Role | Responsibilities |
|------|------|------------------|
|R.L.V.L. MANOHAR | Data Preparation & Backend Development | data preprocessing |
| P. S. V. Ganesh | Dataset Collection & Backend Development | collecting and organizing the dataset |
| M. Moses Pal | Quantum Algorithm Integration & Research and Feature Engineering | integrating quantum algorithms |
| M. Vivekananda Siddhardha | Quantum Algorithm Integration & Research and Feature Engineering | designing feature engineering strategies |
| M. Nasrima Bee | Model Testing & Performance Analysis|  trained models and analyzing performance|
| A. Saranya | Documentation & Presentation |

> Update the table with the actual names and responsibilities of your team members.

---

# Acknowledgements

We express our sincere gratitude to:

- AQVH 2025 Organizing Committee
- Our faculty mentors
- Our institution
- IBM Qiskit Community
- The open-source Python ecosystem

Their guidance and resources were invaluable in completing this project.

---

# License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this project with proper attribution.

---

# Contact

For questions, collaborations, or suggestions, feel free to connect:

**Name:** R.L.V.L. MANOHAR

**Email:** manoharravinuthalalvl@gmail.com

**GitHub:** https://github.com/Manohar4464

---

⭐ **If you found this project helpful, consider giving it a star on GitHub!**
