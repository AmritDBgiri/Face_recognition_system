import json
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_meta():
    with open("models/meta.json", "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    meta = load_meta()
    clf = load("models/classifier_svm.joblib")

    X_test = np.load("models/test_X.npy")
    y_test = np.load("models/test_y.npy")

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("✅ Test Accuracy:", round(acc * 100, 2), "%")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

    # Label names for report
    labelmap = {int(k): v for k, v in meta["labelmap"].items()}
    target_names = [labelmap[i] for i in sorted(labelmap.keys())]

    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
