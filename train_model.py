# ============================================================
# TRAIN MODEL FILE
# Genetic Algorithm vs Grid Search Optimization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import random

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from deap import base, creator, tools, algorithms

# Reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Heart_Disease_Cleaned.csv")
df = df.dropna()

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 2. GENETIC ALGORITHM OPTIMIZATION (WITH CV)
# ============================================================

def evaluate(individual):
    logC, logGamma = individual
    C = 10 ** logC
    gamma = 10 ** logGamma

    model = SVC(C=C, gamma=gamma, kernel='rbf')
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    return (scores.mean(),)

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("logC", random.uniform, -2, 2)
toolbox.register("logGamma", random.uniform, -4, 0)

toolbox.register("individual",
                 tools.initCycle,
                 creator.Individual,
                 (toolbox.logC, toolbox.logGamma),
                 n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=25)
NGEN = 25
CXPB = 0.7
MUTPB = 0.2

fitness_history = []

print("\nRunning Genetic Algorithm Optimization...\n")

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = toolbox.map(toolbox.evaluate, offspring)

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    elite = tools.selBest(population, 1)
    population = toolbox.select(offspring, k=len(population)-1)
    population.extend(elite)

    best = tools.selBest(population, 1)[0]
    fitness_history.append(best.fitness.values[0])

    print(f"Generation {gen+1} Best CV Accuracy: {best.fitness.values[0]*100:.2f}%")

best_individual = tools.selBest(population, 1)[0]
logC, logGamma = best_individual
ga_C = 10 ** logC
ga_gamma = 10 ** logGamma

print("\nBest GA Parameters:")
print("C =", ga_C)
print("Gamma =", ga_gamma)

# Train final GA model
ga_model = SVC(C=ga_C, gamma=ga_gamma, kernel='rbf', probability=True)
ga_model.fit(X_train_scaled, y_train)

ga_test_accuracy = accuracy_score(y_test, ga_model.predict(X_test_scaled))
print("Final GA Test Accuracy:", round(ga_test_accuracy * 100, 2), "%")

# ============================================================
# 3. GRID SEARCH COMPARISON
# ============================================================

print("\nRunning Grid Search...\n")

param_grid = {
    'C': np.logspace(-2, 2, 10),
    'gamma': np.logspace(-4, 0, 10)
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

grid_best = grid.best_estimator_
grid_test_accuracy = accuracy_score(y_test, grid_best.predict(X_test_scaled))

print("Best Grid Parameters:", grid.best_params_)
print("Grid Search Test Accuracy:", round(grid_test_accuracy * 100, 2), "%")

# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, ga_model.predict(X_test_scaled))
print("\nConfusion Matrix (GA Model):")
print(cm)

# ============================================================
# 5. SAVE CONVERGENCE PLOT
# ============================================================

if not os.path.exists("models"):
    os.makedirs("models")

plt.plot(fitness_history)
plt.title("GA Convergence Curve")
plt.xlabel("Generation")
plt.ylabel("Cross Validation Accuracy")
plt.savefig("models/ga_convergence.png")
plt.close()

# ============================================================
# 6. SAVE BEST MODEL
# ============================================================

joblib.dump(ga_model, "models/ga_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\nModel, Scaler, and Feature Columns Saved Successfully!")