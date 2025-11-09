
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier

from utils_metrics import evaluate_and_report

def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['S1','R1','S2','R2','S3','R3','S4','R4','S5','R5']]
    y = df['ORD']
    return X, y

def prep_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_val_s   = sc.transform(X_val)
    X_test_s  = sc.transform(X_test)
    return (X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)

def main(csv_path):
    X, y = load_xy(csv_path)
    X_train_s, y_train, X_val_s, y_val, X_test_s, y_test = prep_splits(X, y)

    models = [
        ("KNN (k=1)", KNeighborsClassifier(n_neighbors=1)),
        ("Random Forest (n=300)", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        # ("XGBoost", XGBClassifier(tree_method='hist', eval_metric='mlogloss', random_state=42)),
    ]

    out = []
    for name, clf in models:
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc, macro, weighted, report = evaluate_and_report(name, y_test, y_pred)
        out.append((name, acc, macro, weighted))
        print(f"\n=== {name} ===\nAccuracy: {acc:.4f}\nMacro F1: {macro:.4f}\nWeighted F1: {weighted:.4f}\n\n{report}\n")
    return out

if __name__ == "__main__":
    # Example: python -m src.run_experiment data/poker_balanced_10k.csv
    results = main("data/poker_balanced_10k.csv")
    import pandas as pd
    pd.DataFrame(results, columns=["Model","Accuracy","Macro F1","Weighted F1"]).to_csv("results/model_comparison.csv", index=False)
    print("Saved results to results/model_comparison.csv")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the saved results
    df = pd.read_csv("results/model_comparison.csv")

    # Plot Weighted F1 Comparison
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="Model", y="Weighted F1", palette="viridis")
    plt.title("Model Comparison — Weighted F1 Scores")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")

    # ✅ Save the figure
    plt.tight_layout()
    plt.savefig("results/model_comparison_f1.png", dpi=300)
    print("✅ Saved F1 comparison graph to results/model_comparison_f1.png")
    plt.close()

