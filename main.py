import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('emails.csv')

df_majority = df[df['spam'] == df['spam'].value_counts().idxmax()]
df_minority = df[df['spam'] == df['spam'].value_counts().idxmin()]

df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_downsampled['text'])
y = df_downsampled['spam']

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
f1_scores = []
precisions = []
recalls = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    accuracies.append(acc)
    f1_scores.append(f1)
    precisions.append(prec)
    recalls.append(rec)

    print(f"Fold {fold}: Accuracy={acc:.3f}, F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

print("\nProsečne metrike za svih 10 foldova:")
print(f"Accuracy: {sum(accuracies)/len(accuracies):.3f}")
print(f"F1 score: {sum(f1_scores)/len(f1_scores):.3f}")
print(f"Precision: {sum(precisions)/len(precisions):.3f}")
print(f"Recall: {sum(recalls)/len(recalls):.3f}")
