import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_balance_data(filepath):
    df = pd.read_csv(filepath)
    df_majority = df[df['spam'] == df['spam'].value_counts().idxmax()]
    df_minority = df[df['spam'] == df['spam'].value_counts().idxmin()]
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_downsampled

def vectorize_text(df, text_column='text'):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[text_column])
    y = df['spam']
    return X, y

def evaluate_model(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precisions, recalls = [], [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

    print("\nProsecne metrike za svih 10 foldova:")
    print(f"Accuracy: {sum(accuracies)/len(accuracies):.3f}")
    print(f"F1 score: {sum(f1_scores)/len(f1_scores):.3f}")
    print(f"Precision: {sum(precisions)/len(precisions):.3f}")
    print(f"Recall: {sum(recalls)/len(recalls):.3f}")

def main():
    df_downsampled = load_and_balance_data('emails.csv')
    X, y = vectorize_text(df_downsampled, text_column='text')

    print("Random Forest:")
    evaluate_model(RandomForestClassifier(random_state=42), X, y, n_splits=10)
    print("\nDecision Tree:")
    evaluate_model(DecisionTreeClassifier(random_state=42), X, y, n_splits=10)
    print("\nLogistic Regression:")
    evaluate_model(LogisticRegression(max_iter=1000, random_state=42), X, y, n_splits=10)
    print("\nNaive Bayes:")
    evaluate_model(MultinomialNB(), X, y, n_splits=10)

if __name__ == "__main__":
    main()
