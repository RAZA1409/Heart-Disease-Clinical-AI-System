# ===============================
# HEART DISEASE OPTIMIZATION PROJECT
# ===============================

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
# ===============================
# 1. LOAD DATASET
# ===============================

df = pd.read_csv("Heart_Disease_Cleaned.csv")

print("Dataset Shape:", df.shape)
print("Target unique values:", df["Heart Disease"].unique())

# Remove any missing values (safety)
df = df.dropna()

# ===============================
# 2. PREPROCESSING
# ===============================

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (Very Important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing Done!")

# ===============================
# 3. BASELINE SVM
# ===============================

baseline_model = SVC()
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("\nBaseline Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

log_experiment(
    method="Baseline SVM",
    accuracy=baseline_accuracy,
    params={}
)
# ===============================
# 4. GRID SEARCH
# ===============================

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

best_model = grid.best_estimator_

y_pred_grid = best_model.predict(X_test)

grid_accuracy = accuracy_score(y_test, y_pred_grid)

print("Grid Search Accuracy:",
      round(grid_accuracy * 100, 2), "%")


log_experiment(
    method="Grid Search SVM",
    accuracy=grid_accuracy,
    params=grid.best_params_
)

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(best_model, "models/best_model.pkl")
# ===============================
# 5. COMPARISON
# ===============================

print("\n---- FINAL COMPARISON ----")
print("Baseline Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

print("Grid Search Accuracy:",
      round(grid_accuracy * 100, 2), "%")



# ===============================
# 6. GENETIC ALGORITHM (DEAP)
# ===============================

from deap import base, creator, tools, algorithms
import random

print("\nStarting Genetic Algorithm Optimization...")

# Fitness function
def evaluate(individual):
    C = abs(individual[0])
    gamma = abs(individual[1])

    # Prevent extremely small values
    if C < 0.001:
        C = 0.001
    if gamma < 0.0001:
        gamma = 0.0001

    model = SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    return (acc,)

# Avoid recreation error if rerun
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("C", random.uniform, 0.1, 100)
toolbox.register("gamma", random.uniform, 0.0001, 1)

toolbox.register("individual",
                 tools.initCycle,
                 creator.Individual,
                 (toolbox.C, toolbox.gamma),
                 n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# GA Parameters
population = toolbox.population(n=10)
NGEN = 10
CXPB = 0.7
MUTPB = 0.2

fitness_history = []

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

    fits = toolbox.map(toolbox.evaluate, offspring)

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

    best = tools.selBest(population, 1)[0]
    fitness_history.append(best.fitness.values[0])

    print(f"Generation {gen+1} Best Accuracy: {best.fitness.values[0]*100:.2f}%")

best_individual = tools.selBest(population, 1)[0]

ga_C = abs(best_individual[0])
ga_gamma = abs(best_individual[1])

if ga_C < 0.001:
    ga_C = 0.001
if ga_gamma < 0.0001:
    ga_gamma = 0.0001

print("\nBest GA Parameters:")
print("C =", ga_C)
print("gamma =", ga_gamma)

ga_model = SVC(C=ga_C, gamma=ga_gamma, kernel='rbf')
ga_model.fit(X_train, y_train)

y_pred_ga = ga_model.predict(X_test)
ga_accuracy = accuracy_score(y_test, y_pred_ga)

print("GA Accuracy:", round(ga_accuracy * 100, 2), "%")

# Log GA experiment
log_experiment(
    method="Genetic Algorithm SVM",
    accuracy=ga_accuracy,
    params={"C": ga_C, "gamma": ga_gamma, "kernel": "rbf"}
)

# Plot convergence
plt.figure()
plt.plot(range(1, NGEN+1), fitness_history)
plt.title("GA Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Accuracy")
plt.show()