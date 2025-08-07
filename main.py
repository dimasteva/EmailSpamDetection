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

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

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

def evaluate_model(model, X, y, df, n_splits=10, vectorizer=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precisions, recalls, k_words = [], [], [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

          # SHAP analiza
        if (isinstance(model, RandomForestClassifier) or isinstance(model, DecisionTreeClassifier)) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance_rf_dt(model, X_train, feature_names)
            model_name = type(model).__name__.replace("Classifier", "").replace("RandomForest", "Random Forest").replace("DecisionTree", "Decision Tree")
            print(f"\nTop 20 najvaznijih reci po SHAP znacaju ({model_name}):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {val:.5f}")

            top_words = [word for word, _ in shap_importance]
            train_texts = df.iloc[train_index]['text']
            test_texts = df.iloc[test_index]['text']
            k = find_k_for_target_accuracy(model.__class__, train_texts, test_texts, y_train, y_test, top_words)
            print(f"[Fold {fold}] K = {k} reci je dovoljno za 80% tacnosti")
            k_words.append(k)

        elif isinstance(model, LogisticRegression) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance(model, X_train, feature_names)
            print("\nTop 20 najvaznijih reci po SHAP znacaju (Logistic Regression):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {val:.5f}")

            top_words = [word for word, _ in shap_importance]
            train_texts = df.iloc[train_index]['text']
            test_texts = df.iloc[test_index]['text']
            k = find_k_for_target_accuracy(model.__class__, train_texts, test_texts, y_train, y_test, top_words)
            print(f"[Fold {fold}] K = {k} reci je dovoljno za 80% tacnosti")
            k_words.append(k)

        elif isinstance(model, MultinomialNB) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance_nb(model, X_train, feature_names)
            print("\nTop 20 najvaznijih reci po SHAP znacaju (Naive Bayes):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {float(val[0]):.5f}")
            
            top_words = [word for word, _ in shap_importance]
            train_texts = df.iloc[train_index]['text']
            test_texts = df.iloc[test_index]['text']
            k = find_k_for_target_accuracy(model.__class__, train_texts, test_texts, y_train, y_test, top_words)
            print(f"[Fold {fold}] K = {k} reci je dovoljno za 80% tacnosti")
            k_words.append(k)

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

    return f1_scores, k_words

def compute_shap_importance(model, X_sample, feature_names):
    explainer = shap.Explainer(model, X_sample, feature_names=feature_names)
    shap_values = explainer(X_sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance

def compute_shap_importance_rf_dt(model, X_sample, feature_names):
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

def compute_shap_importance_nb(model, X_sample, feature_names):
    X_sample_dense = X_sample.toarray() if hasattr(X_sample, "toarray") else X_sample

    background = shap.sample(X_sample_dense, 50, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X_sample_dense[:20], nsamples=100)

    shap_vals = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

    shap_vals = np.array(shap_vals)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    shap_importance_list = list(zip(feature_names, mean_abs_shap.tolist()))
    shap_importance_list.sort(key=lambda x: x[1], reverse=True)

    return shap_importance_list

def find_k_for_target_accuracy(model_class, train_texts, test_texts, y_train, y_test, top_words, accuracy_threshold=0.8):
    for k in range(1, len(top_words) + 1):
        selected_words = top_words[:k]
        vectorizer = TfidfVectorizer(vocabulary=selected_words)
        X_train_k = vectorizer.fit_transform(train_texts)
        X_test_k = vectorizer.transform(test_texts)

        model = model_class()
        model.fit(X_train_k, y_train)
        y_pred = model.predict(X_test_k)
        acc = accuracy_score(y_test, y_pred)
        print(f"K={k}, Accuracy={acc:.3f}")

        if acc >= accuracy_threshold:
            return k
    return None

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

def evaluate_bilstm(X, y, tokenizer, df, n_splits=10, epochs=3, batch_size=32, max_words=5000, max_len=100):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precisions, recalls, k_words = [], [], [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = build_bilstm_model(max_words=max_words, max_len=max_len)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        try:
            background = X_train[:50].astype('float32')
            test_sample = X_test[:10].astype('float32')

            explainer = shap.KernelExplainer(lambda x: model.predict(x).flatten(), background)
            shap_values = explainer.shap_values(test_sample, nsamples=100)

            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_matrix = shap_values[1]
            else:
                shap_matrix = shap_values

            if shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, 0, :]

            shap_matrix = np.abs(shap_matrix)

            if shap_matrix.shape != test_sample.shape:
                print(f"[Fold {fold}] Oblik SHAP matrice {shap_matrix.shape} se ne poklapa sa ulazom {test_sample.shape}")
            else:
                word_shap_scores = np.zeros(max_words)
                word_counts = np.zeros(max_words)

                for i in range(test_sample.shape[0]):
                    for j in range(test_sample.shape[1]):
                        word_id = int(test_sample[i][j])
                        if word_id != 0 and word_id < max_words:
                            word_shap_scores[word_id] += shap_matrix[i][j]
                            word_counts[word_id] += 1

                mean_shap = word_shap_scores / (word_counts + 1e-8)
                index_word = {v: k for k, v in tokenizer.word_index.items()}
                used_word_scores = [(idx, mean_shap[idx]) for idx in range(1, max_words) if word_counts[idx] > 0]
                used_word_scores = sorted(used_word_scores, key=lambda x: x[1], reverse=True)

                print(f"\nTop 20 reci po SHAP znacaju (Fold {fold}):")
                for idx, score in used_word_scores[:20]:
                    word = index_word.get(idx, 'UNK')
                    print(f"{word:<15} | SHAP: {score:.6f}")

        except Exception as e:
            print(f"[Fold {fold}] SHAP analiza nije uspela: {e}")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        accuracies.append(acc)
        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)

        top_words = [index_word[idx] for idx, _ in used_word_scores]
        train_texts = [df['text'].iloc[i] for i in train_index]
        test_texts = [df['text'].iloc[i] for i in test_index]

        k = find_k_for_bilstm(train_texts, test_texts, y_train, y_test, top_words, accuracy_threshold=0.8)
        print(f"[Fold {fold}] K = {k} reci je dovoljno za 80% tacnosti")
        k_words.append(k)

        print(f"Rezultati (Fold {fold}): Accuracy={acc:.3f}, F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    print("\n=== Prosecne metrike za svih 10 foldova (BiLSTM) ===")
    print(f"Accuracy:  {np.mean(accuracies):.3f}")
    print(f"F1 score:  {np.mean(f1_scores):.3f}")
    print(f"Precision: {np.mean(precisions):.3f}")
    print(f"Recall:    {np.mean(recalls):.3f}")

    return f1_scores, k_words

def find_k_for_bilstm(train_texts, test_texts, y_train, y_test, top_words, accuracy_threshold=0.8, max_len=100):
    for k in range(1, len(top_words) + 1):
        selected_words = top_words[:k]
        word_to_index = {word: i+1 for i, word in enumerate(selected_words)}

        def texts_to_sequences(texts):
            sequences = []
            for text in texts:
                words = re.findall(r'\b\w+\b', text.lower())
                sequence = [word_to_index[word] for word in words if word in word_to_index]
                sequences.append(sequence)
            return sequences

        X_train_seq = pad_sequences(texts_to_sequences(train_texts), maxlen=max_len)
        X_test_seq = pad_sequences(texts_to_sequences(test_texts), maxlen=max_len)

        model = build_bilstm_model(max_words=k+1, max_len=max_len)
        model.fit(X_train_seq, y_train, epochs=3, batch_size=32, verbose=0)

        y_pred_prob = model.predict(X_test_seq)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        print(f"K={k}, Accuracy={acc:.3f}")

        if acc >= accuracy_threshold:
            return k
    return None



def main():
    df_downsampled = load_and_balance_data('emails.csv')
    df_downsampled['text'] = df_downsampled['text'].apply(remove_stopwords)

    results = []

    print("\n=== Bi-LSTM ===")
    max_words = 5000
    max_len = 100
    X_bilstm, y_bilstm, tokenizer = prepare_bilstm_data(df_downsampled, text_column='text', max_words=max_words, max_len=max_len)
    f1_bilstm, k_bilstm = evaluate_bilstm(X_bilstm, y_bilstm, tokenizer, df_downsampled, n_splits=10, epochs=3, batch_size=32, max_words=max_words, max_len=max_len)
    results.append(('BiLSTM', f1_bilstm, k_bilstm))

    X, y, vectorizer = vectorize_text(df_downsampled, text_column='text')

    print("\n--- Random Forest ---")
    f1_rf, k_rf = evaluate_model(RandomForestClassifier(random_state=42), X, y, df_downsampled, n_splits=10, vectorizer=vectorizer)
    results.append(('Random Forest', f1_rf, k_rf))

    print("\n--- Decision Tree ---")
    f1_dt, k_dt = evaluate_model(DecisionTreeClassifier(random_state=42), X, y, df_downsampled, n_splits=10, vectorizer=vectorizer)
    results.append(('Decision Tree', f1_dt, k_dt))

    print("\n--- Logistic Regression ---")
    f1_lr, k_lr = evaluate_model(LogisticRegression(max_iter=1000, random_state=42), X, y, df_downsampled, n_splits=10, vectorizer=vectorizer)
    results.append(('Logistic Regression', f1_lr, k_lr))

    print("\n--- Naive Bayes ---")
    f1_nb, k_nb = evaluate_model(MultinomialNB(), X, y, df_downsampled, n_splits=10, vectorizer=vectorizer)
    results.append(('Naive Bayes', f1_nb, k_nb))

    summary_data = []
    for model_name, f1_list, k_list in results:
        for fold, (f1, k) in enumerate(zip(f1_list, k_list), 1):
            summary_data.append({
                'Model': model_name,
                'Fold': fold,
                'F1 Score': round(f1, 4),
                'K Words': k
            })

    summary_df = pd.DataFrame(summary_data)
    print("\n=== Rezime svih modela ===")
    print(summary_df)

    summary_df.to_csv("results_summary.csv", index=False)
    print("\nRezultati su sacuvani u 'results_summary.csv'.")


if __name__ == "__main__":
    main()
