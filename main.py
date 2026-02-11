# ============================================================
# HEART DISEASE OPTIMIZATION PROJECT
# Objective:
# Compare Baseline SVM, Grid Search, and Genetic Algorithm
# for Hyperparameter Optimization.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from experiment_logger import log_experiment
import joblib
import os


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Load cleaned heart disease dataset
df = pd.read_csv("Heart_Disease_Cleaned.csv")

print("Dataset Shape:", df.shape)
print("Target unique values:", df["Heart Disease"].unique())

# Remove any missing values (safety step)
df = df.dropna()


# ============================================================
# 2. PREPROCESSING
# ============================================================

# Separate features and target
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Split dataset into training (80%) and testing (20%)
# Stratified split maintains class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (SVM is sensitive to feature scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit only on training data
X_test = scaler.transform(X_test)        # Apply same scaling to test data

print("Preprocessing Done!")


# ============================================================
# 3. BASELINE SVM (No Hyperparameter Tuning)
# ============================================================

# Train default SVM model
baseline_model = SVC()
baseline_model.fit(X_train, y_train)

# Predict on test set
y_pred_baseline = baseline_model.predict(X_test)

# Evaluate accuracy
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("\nBaseline Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

# Log experiment result
log_experiment(
    method="Baseline SVM",
    accuracy=baseline_accuracy,
    params={}
)


# ============================================================
# 4. GRID SEARCH (Exhaustive Hyperparameter Search)
# ============================================================

# Define search space
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Perform 5-fold cross-validation grid search
grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Train grid search
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

# Get best model found
best_model = grid.best_estimator_

# Evaluate best model
y_pred_grid = best_model.predict(X_test)
grid_accuracy = accuracy_score(y_test, y_pred_grid)

print("Grid Search Accuracy:",
      round(grid_accuracy * 100, 2), "%")

# Log experiment
log_experiment(
    method="Grid Search SVM",
    accuracy=grid_accuracy,
    params=grid.best_params_
)

# Save best model to disk
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(best_model, "models/best_model.pkl")


# ============================================================
# 5. FINAL COMPARISON
# ============================================================

print("\n---- FINAL COMPARISON ----")
print("Baseline Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

print("Grid Search Accuracy:",
      round(grid_accuracy * 100, 2), "%")


# ============================================================
# 6. GENETIC ALGORITHM (DEAP)
# Population-based evolutionary optimization
# ============================================================

from deap import base, creator, tools, algorithms
import random

print("\nStarting Genetic Algorithm Optimization...")


# ------------------------------------------------------------
# Fitness Function
# Evaluates accuracy of SVM using log-scale hyperparameters
# ------------------------------------------------------------
def evaluate(individual):
    logC, logGamma = individual

    # Convert log-scale values back to real hyperparameters
    C = 10 ** logC
    gamma = 10 ** logGamma

    model = SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    return (acc,)


# Avoid recreation errors if script is re-run
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Log-scale search space:
# logC ∈ [-2, 2]  → C ∈ [0.01, 100]
# logGamma ∈ [-4, 0] → gamma ∈ [0.0001, 1]
toolbox.register("logC", random.uniform, -2, 2)
toolbox.register("logGamma", random.uniform, -4, 0)

toolbox.register("individual",
                 tools.initCycle,
                 creator.Individual,
                 (toolbox.logC, toolbox.logGamma),
                 n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# GA operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)        # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection


# ------------------------------------------------------------
# GA Hyperparameters
# ------------------------------------------------------------
population = toolbox.population(n=20)  # Population size
NGEN = 20                              # Number of generations
CXPB = 0.7                             # Crossover probability
MUTPB = 0.2                            # Mutation probability

fitness_history = []

# ------------------------------------------------------------
# Evolution Process
# ------------------------------------------------------------
for gen in range(NGEN):

    # Apply crossover and mutation
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

    # Evaluate offspring
    fits = toolbox.map(toolbox.evaluate, offspring)

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    # Elitism: keep best individual from previous generation
    elite = tools.selBest(population, 1)
    population = toolbox.select(offspring, k=len(population)-1)
    population.extend(elite)

    # Track best accuracy per generation
    best = tools.selBest(population, 1)[0]
    fitness_history.append(best.fitness.values[0])

    print(f"Generation {gen+1} Best Accuracy: {best.fitness.values[0]*100:.2f}%")


# ------------------------------------------------------------
# Extract Best Individual
# ------------------------------------------------------------
best_individual = tools.selBest(population, 1)[0]
logC, logGamma = best_individual

ga_C = 10 ** logC
ga_gamma = 10 ** logGamma

print("\nBest GA Parameters:")
print("C =", ga_C)
print("gamma =", ga_gamma)

# Train final GA-optimized model
ga_model = SVC(C=ga_C, gamma=ga_gamma, kernel='rbf')
ga_model.fit(X_train, y_train)

y_pred_ga = ga_model.predict(X_test)
ga_accuracy = accuracy_score(y_test, y_pred_ga)

print("GA Accuracy:", round(ga_accuracy * 100, 2), "%")

# Log experiment
log_experiment(
    method="Genetic Algorithm SVM",
    accuracy=ga_accuracy,
    params={"C": ga_C, "gamma": ga_gamma, "kernel": "rbf"}
)


# ------------------------------------------------------------
# Plot GA Convergence
# ------------------------------------------------------------
plt.figure()
plt.plot(range(1, NGEN+1), fitness_history)
plt.title("GA Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Accuracy")
plt.show()