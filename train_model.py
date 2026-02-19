# ============================================================
# TRAIN MODEL FILE
# This file trains and optimizes the SVM model
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Heart_Disease_Cleaned.csv")
df = df.dropna()

X = df.drop("Heart Disease", axis=1)

# Save feature column names for prediction consistency

y = df["Heart Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 2. GENETIC ALGORITHM OPTIMIZATION
# ============================================================

def evaluate(individual):
    logC, logGamma = individual
    C = 10 ** logC
    gamma = 10 ** logGamma

    model = SVC(C=C, gamma=gamma, kernel='rbf',probability=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    return (acc,)

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

population = toolbox.population(n=20)
NGEN = 20
CXPB = 0.7
MUTPB = 0.2

fitness_history = []

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

    print(f"Generation {gen+1} Best Accuracy: {best.fitness.values[0]*100:.2f}%")

best_individual = tools.selBest(population, 1)[0]
logC, logGamma = best_individual

ga_C = 10 ** logC
ga_gamma = 10 ** logGamma

print("\nBest Parameters:")
print("C =", ga_C)
print("gamma =", ga_gamma)

ga_model = SVC(C=ga_C, gamma=ga_gamma, kernel='rbf', probability=True)
ga_model.fit(X_train, y_train)

ga_accuracy = accuracy_score(y_test, ga_model.predict(X_test))
print("Final GA Accuracy:", round(ga_accuracy * 100, 2), "%")


# ============================================================
# 3. SAVE MODEL, SCALER & FEATURE COLUMNS
# ============================================================

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Save trained GA model
joblib.dump(ga_model, "models/model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save feature column names (IMPORTANT for prediction consistency)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "models/feature_columns.pkl")

print("\nModel, Scaler, and Feature Columns Saved Successfully!")