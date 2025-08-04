import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import shap
import matplotlib.pyplot as plt

def load_and_balance_data(filepath):
    df = pd.read_csv(filepath)
    df_majority = df[df['spam'] == df['spam'].value_counts().idxmax()]
    df_minority = df[df['spam'] == df['spam'].value_counts().idxmin()]
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_downsampled

def vectorize_text(df, text_column='text'):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[text_column])
    y = df['spam']
    return X, y, vectorizer

def evaluate_model(model, X, y, n_splits=10, vectorizer=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precisions, recalls = [], [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

          # SHAP analiza
        if (isinstance(model, RandomForestClassifier) or isinstance(model, DecisionTreeClassifier)) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance_rf(model, X_train, feature_names)
            model_name = type(model).__name__.replace("Classifier", "").replace("RandomForest", "Random Forest").replace("DecisionTree", "Decision Tree")
            print(f"\nTop 20 najvaznijih reci po SHAP znacaju ({model_name}):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {val:.5f}")
        elif isinstance(model, LogisticRegression) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance(model, X_train, feature_names)
            print("\nTop 20 najvaznijih reci po SHAP znacaju (Logistic Regression):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {val:.5f}")

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

def compute_shap_importance(model, X_sample, feature_names):
    explainer = shap.Explainer(model, X_sample, feature_names=feature_names)
    shap_values = explainer(X_sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance

def compute_shap_importance_rf(model, X_sample, feature_names):
    X_sample_dense = X_sample.toarray() if hasattr(X_sample, "toarray") else X_sample
    
    background = shap.sample(X_sample_dense, 50, random_state=42)
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")

    shap_values = explainer.shap_values(X_sample_dense)
    shap_values_to_use = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

    mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)

    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=1)

    shap_importance_list = list(zip(feature_names, mean_abs_shap))
    shap_importance_list.sort(key=lambda x: x[1], reverse=True)
    
    return shap_importance_list

def prepare_bilstm_data(df, text_column='text', max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df[text_column])
    sequences = tokenizer.texts_to_sequences(df[text_column])
    X = pad_sequences(sequences, maxlen=max_len)
    y = df['spam'].values
    return X, y, tokenizer

def build_bilstm_model(max_words=5000, max_len=100):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate_bilstm(X, y, n_splits=10, epochs=3, batch_size=32, max_words=5000, max_len=100):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precisions, recalls = [], [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = build_bilstm_model(max_words=max_words, max_len=max_len)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        accuracies.append(acc)
        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)

        print(f"Fold {fold}: Accuracy={acc:.3f}, F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    print("\nProsecne metrike za svih 10 foldova (BiLSTM):")
    print(f"Accuracy: {sum(accuracies)/len(accuracies):.3f}")
    print(f"F1 score: {sum(f1_scores)/len(f1_scores):.3f}")
    print(f"Precision: {sum(precisions)/len(precisions):.3f}")
    print(f"Recall: {sum(recalls)/len(recalls):.3f}")

def main():
    df_downsampled = load_and_balance_data('emails.csv')
    X, y, vectorizer = vectorize_text(df_downsampled, text_column='text')

    #print("\nBi-LSTM:")
    #max_words = 5000
    #max_len = 100
    #X_bilstm, y_bilstm, _ = prepare_bilstm_data(df_downsampled, text_column='text', max_words=max_words, max_len=max_len)
    #evaluate_bilstm(X_bilstm, y_bilstm, n_splits=10, epochs=3, batch_size=32, max_words=max_words, max_len=max_len)

    print("Random Forest:")
    evaluate_model(RandomForestClassifier(random_state=42), X, y, n_splits=10, vectorizer=vectorizer)

    print("\nDecision Tree:")
    evaluate_model(DecisionTreeClassifier(random_state=42), X, y, n_splits=10, vectorizer=vectorizer)

    print("\nLogistic Regression:")
    evaluate_model(LogisticRegression(max_iter=1000, random_state=42), X, y, n_splits=10, vectorizer=vectorizer)

    #print("\nNaive Bayes:")
    #evaluate_model(MultinomialNB(), X, y, n_splits=10)



if __name__ == "__main__":
    main()
