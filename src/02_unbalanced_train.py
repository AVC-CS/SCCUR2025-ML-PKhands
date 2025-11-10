# ============================
# 02_unbalanced_train.py — Save bar chart as PNG
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os


def main():
    print("=== Training on UNBALANCED dataset ===")

    # ----------------------------
    # Load and prepare data
    # ----------------------------
    df = pd.read_csv("../data/poker-hand-training-true.data", header=None)
    df.columns = ['S1', 'R1', 'S2', 'R2', 'S3',
                  'R3', 'S4', 'R4', 'S5', 'R5', 'ORD']
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    X = df[['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5']]
    y = df['ORD']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ----------------------------
    # Define and evaluate models
    # ----------------------------
    models = [
        ("KNN (k=1)", KNeighborsClassifier(n_neighbors=1)),
        ("Random Forest (n=300)", RandomForestClassifier(
            n_estimators=300, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("XGBoost", XGBClassifier(tree_method='hist',
         eval_metric='mlogloss', random_state=42))
    ]

    model_results = pd.DataFrame(
        columns=["Model", "Accuracy", "Macro F1", "Weighted F1"])

    for name, clf in models:
        print(f"\n=== Evaluating {name} ===")
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted = f1_score(
            y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(
            y_test, y_pred, digits=3, zero_division=0)

        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {macro:.4f}")
        print(f"Weighted F1: {weighted:.4f}")
        print(report)

        model_results.loc[len(model_results)] = [name, acc, macro, weighted]

    # ----------------------------
    # Save numeric results
    # ----------------------------
    out_csv = os.path.join(os.path.dirname(__file__),
                           "unbalanced_results_auto.csv")
    model_results.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to {out_csv}")

    # ----------------------------
    # Create and save bar graph (Accuracy / F1)
    # ----------------------------
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    model_results_melted = model_results.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1", "Weighted F1"],
                                              var_name="Metric", value_name="Score")
    sns.barplot(data=model_results_melted, x="Model",
                y="Score", hue="Metric", palette="Set2")
    plt.title("Model Performance — Unbalanced Dataset")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(__file__),
                           "unbalanced_results_bar.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"📊 Bar graph saved to {out_png}")
    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
