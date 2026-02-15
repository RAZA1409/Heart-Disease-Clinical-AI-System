# 🫀 Heart Disease Clinical AI System (GA-Optimized Machine Learning Model)

### Intelligent Clinical Risk Prediction Platform with Genetic Algorithm Based Hyperparameter Optimization

An end-to-end AI-powered clinical decision support system that predicts heart disease risk using a **Genetic Algorithm (GA) optimized Support Vector Machine (SVM)** model.

This system integrates:

- 🧠 Machine Learning Model Optimization
- 🔬 Evolutionary Algorithm (Genetic Algorithm)
- 🌐 Full-Stack Flask Web Application
- 🔐 Secure Authentication System
- 📊 Clinical Dashboard & Risk Analytics
- 📁 Patient Record Management

The project simulates a real-world hospital clinical environment and demonstrates how optimization techniques can enhance predictive performance in healthcare AI systems.

## 🚀 Features

- 🔬 Genetic Algorithm based Hyperparameter Optimization
- 🤖 Optimized SVM Model for Heart Disease Prediction
- 🌐 Interactive Flask Web Interface
- 🔐 Secure Login & Session Management
- 🔑 Password Change System
- 🆔 Automatic Patient ID Generation
- 📊 Risk Classification (Low / Moderate / High)
- 📁 Patient Record Storage & History Management
- 🗑 Record Deletion Functionality
- 🧠 Modular ML Architecture (Training & Prediction separated)

---

## 🏗 Project Architecture

Heart-Disease-Clinical-AI-System/
│
├── app.py                     # Flask Web Application
├── auth.py                    # Authentication & Database Setup
├── train_model.py             # Model training + GA optimization
├── predict.py                 # Prediction logic
├── experiment_logger.py       # Logs model experiments
├── database.db                # SQLite authentication database
├── records/                   # Patient history storage
├── templates/                 # HTML templates
├── static/                    # CSS styling
├── models/                    # Saved ML models
└── requirements.txt           # Project dependencies

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
- Genetic Algorithm (Custom Evolutionary Optimization)
- SQLite (Authentication Database)
- Joblib (Model Serialization)
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

## 🔐 Authentication & Security

- Secure password hashing using Werkzeug
- Session-based authentication
- Protected routes (Dashboard, History, Assessment)
- Password change functionality
- Confirmation before record deletion


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

http://127.0.0.1:5000/login

---

## 🎯 System Use Case

This project can be used for:

- Academic demonstration of ML + Optimization integration
- Healthcare risk prediction simulation
- Clinical decision-support prototype
- Placement-ready AI portfolio project

---

## 🔮 Future Enhancements

- 🗄 Migration from CSV to Full SQL Database
- 📄 PDF Medical Report Export
- 📊 Interactive Risk Distribution Charts
- 🧑‍⚕️ Role-Based Access Control (Admin / Doctor)
- ☁ Cloud Deployment (Render / AWS)
- 🔒 Advanced Security (Session Timeout, Rate Limiting)
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