# 🫀 Heart Disease Clinical AI System  
### GA-Optimized SVM + Flask Web Application

An end-to-end AI-powered healthcare risk prediction system that uses **Genetic Algorithm (GA)** to optimize **Support Vector Machine (SVM)** hyperparameters and provides predictions through a professional **Flask web interface**.

This project simulates a real-world clinical decision support system for heart disease risk assessment and demonstrates the integration of Machine Learning, Optimization Techniques, and Full-Stack Development.

---

## 🚀 Features

- 🔬 Genetic Algorithm based Hyperparameter Optimization
- 🤖 Optimized SVM Model for Heart Disease Prediction
- 🌐 Interactive Flask Web Interface
- 🆔 Automatic Patient ID Generation
- 📊 Risk Classification (Low / Medium / High)
- 📁 Patient Record Storage (CSV-based)
- 📈 Experiment Logging for Model Comparison
- 🧠 Modular ML Architecture (Training & Prediction separated)

---

## 🏗 Project Architecture

```
Heart-Disease-Clinical-AI-System/
│
├── app.py                     # Flask Web Application
├── train_model.py             # Model training + GA optimization
├── predict.py                 # Prediction logic
├── main.py                    # Main execution script
├── experiment_logger.py       # Logs model experiments
├── results.csv                # Stored patient predictions
├── Heart_Disease_Cleaned.csv  # Cleaned dataset
├── templates/                 # HTML templates
├── static/                    # CSS styling
└── requirements.txt           # Project dependencies
```

---

## 🧠 Machine Learning Workflow

1. Data Cleaning & Preprocessing  
2. Baseline SVM Model Creation  
3. Hyperparameter Optimization using Genetic Algorithm  
4. Cross-Validation & Performance Evaluation  
5. Integration into Flask Web Application  

---

## 📊 Technologies Used

- Python
- Scikit-learn
- Genetic Algorithm
- Flask
- HTML / CSS
- Pandas
- NumPy

---

## 📈 Optimization Strategy

Instead of traditional Grid Search, this system uses a **Genetic Algorithm (GA)** approach:

- Population-based search mechanism
- Fitness evaluation using cross-validation accuracy
- Selection, crossover, and mutation operations
- Best-performing hyperparameters selected automatically

This approach improves search efficiency and explores a broader solution space compared to exhaustive grid search.

---

## 🖥 How To Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/RAZA1409/Heart-Disease-Clinical-AI-System.git
cd Heart-Disease-Clinical-AI-System
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model (Optional)

```bash
python train_model.py
```

### 4️⃣ Run Flask Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🎯 System Use Case

This project can be used for:

- Academic demonstration of ML + Optimization integration
- Healthcare risk prediction simulation
- Clinical decision-support prototype
- Placement-ready AI portfolio project

---

## 🔮 Future Enhancements

- 🔐 User Authentication System
- 🗄 SQL Database Integration (SQLite / MySQL)
- 📄 PDF Medical Report Generation
- 📊 Interactive Analytics Dashboard
- ☁ Cloud Deployment (Render / AWS)

---

## 👨‍💻 Author

**Mohammad Raza**  
B.Tech CSE (AI & ML)  
Lovely Professional University  

---

## ⭐ Project Strength

✔ Combines Machine Learning + Optimization  
✔ Includes Web Deployment  
✔ Healthcare Domain Application  
✔ Modular & Scalable Structure  
✔ Strong Placement-Level AI Project  

---

### 🚀 Built with Passion for AI & Real-World Impact