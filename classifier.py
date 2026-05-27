import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset
import joblib
import os

MODEL_PATH = "models/classifier.pkl"

def train():
    """
    Dataset: https://huggingface.co/datasets/deepset/prompt-injections
    Download CSV and place in data/dataset.csv
    Columns expected: 'text' (prompt), 'label' (0=safe, 1=injection)
    """
    ds = load_dataset("deepset/prompt-injections")
    df = ds["train"].to_pandas()
    df = df.dropna(subset=["text", "label"])
 
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams catch phrase patterns
            max_features=10000,
            sublinear_tf=True     # reduces impact of very frequent terms
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced"  # handles imbalanced datasets
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["safe", "injection"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def predict(text: str) -> dict:
    """Returns label, confidence score, and raw probabilities."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train() first.")

    pipeline = joblib.load(MODEL_PATH)
    proba = pipeline.predict_proba([text])[0]
    label = int(pipeline.predict([text])[0])
    confidence = float(max(proba))

    return {
        "label": label,           # 0 = safe, 1 = injection
        "label_text": "INJECTION" if label == 1 else "SAFE",
        "confidence": confidence,
        "prob_safe": float(proba[0]),
        "prob_injection": float(proba[1])
    }

if __name__ == "__main__":
    train()