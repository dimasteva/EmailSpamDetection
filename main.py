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
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

import shap
import matplotlib.pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

def remove_stopwords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

def load_and_balance_data(filepath, downsample=False):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if not downsample:
        return df
    df_majority = df[df['spam'] == df['spam'].value_counts().idxmax()]
    df_minority = df[df['spam'] == df['spam'].value_counts().idxmin()]
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_downsampled

def evaluate_model(model, df, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precisions, recalls, k_words, top_words_list = [], [], [], [], [], []

    # Ucitamo emails.csv samo jednom
    #emails_df = pd.read_csv("novi2.csv")
    #emails_texts = emails_df['text']
    #emails_labels = emails_df['spam']

    for fold, (train_index, test_index) in enumerate(skf.split(df['text'], df['spam']), 1):
        train_texts = df.iloc[train_index]['text']
        test_texts = df.iloc[test_index]['text']
        y_train = df.iloc[train_index]['spam']
        y_test = df.iloc[test_index]['spam']

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            norm='l2'
        )

        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if (isinstance(model, RandomForestClassifier) or isinstance(model, DecisionTreeClassifier)) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance_rf_dt(model, X_train, X_test, feature_names)
            model_name = type(model).__name__.replace("Classifier", "").replace("RandomForest", "Random Forest").replace("DecisionTree", "Decision Tree")
            print(f"\nTop 20 najvaznijih reci po SHAP znacaju ({model_name}):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {val:.5f}")
            top_20_words = [word for word, _ in shap_importance[:20]]
            top_words_list.append(top_20_words)

            top_words = [word for word, _ in shap_importance]
            train_texts = df.iloc[train_index]['text']
            test_texts = df.iloc[test_index]['text']
            k = find_k_for_target_accuracy(model, train_texts, test_texts, y_train, y_test, top_words)
            print(f"[Fold {fold}] K = {k} reci je dovoljno za 80% tacnosti")
            k_words.append(k)

        elif isinstance(model, LogisticRegression) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance(model, X_train, X_test, feature_names)
            print("\nTop 20 najvaznijih reci po SHAP znacaju (Logistic Regression):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {val:.5f}")
            top_20_words = [word for word, _ in shap_importance[:20]]
            top_words_list.append(top_20_words)

            top_words = [word for word, _ in shap_importance]
            train_texts = df.iloc[train_index]['text']
            test_texts = df.iloc[test_index]['text']
            k = find_k_for_target_accuracy(model, train_texts, test_texts, y_train, y_test, top_words)
            print(f"[Fold {fold}] K = {k} reci je dovoljno za 80% tacnosti")
            k_words.append(k)

        elif isinstance(model, MultinomialNB) and vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            shap_importance = compute_shap_importance_nb(model, X_train, X_test, feature_names)
            print("\nTop 20 najvaznijih reci po SHAP znacaju (Naive Bayes):")
            for word, val in shap_importance[:20]:
                print(f"{word:<20} SHAP: {float(val[0]):.5f}")
            top_20_words = [word for word, _ in shap_importance[:20]]
            top_words_list.append(top_20_words)

            top_words = [word for word, _ in shap_importance]
            train_texts = df.iloc[train_index]['text']
            test_texts = df.iloc[test_index]['text']
            k = find_k_for_target_accuracy(model, train_texts, test_texts, y_train, y_test, top_words)
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

    return f1_scores, k_words, top_words_list

def compute_shap_importance(model, X_train, X_test, feature_names, background_size=None, n_test=None):
    # --- background set ---
    if background_size is None or background_size >= X_train.shape[0]:
        background = X_train.astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        idx_bg = rng.choice(X_train.shape[0], background_size, replace=False)
        background = X_train[idx_bg].astype(np.float32)

    explainer = shap.Explainer(model, background, feature_names=feature_names)

    # --- test sample ---
    if n_test is None or n_test >= X_test.shape[0]:
        X_test_sample = X_test
    else:
        rng = np.random.default_rng(42)
        idx_test = rng.choice(X_test.shape[0], n_test, replace=False)
        X_test_sample = X_test[idx_test]

    shap_values = explainer(X_test_sample)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance


def compute_shap_importance_rf_dt(model, X_train, X_test, feature_names, background_size=None, n_test=None):
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    # --- background set ---
    if background_size is None or background_size >= X_train_dense.shape[0]:
        background = X_train_dense.astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        idx_bg = rng.choice(X_train_dense.shape[0], background_size, replace=False)
        background = X_train_dense[idx_bg].astype(np.float32)

    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")

    # --- test sample ---
    if n_test is None or n_test >= X_test_dense.shape[0]:
        X_test_sample = X_test_dense
    else:
        rng = np.random.default_rng(42)
        idx_test = rng.choice(X_test_dense.shape[0], n_test, replace=False)
        X_test_sample = X_test_dense[idx_test]

    shap_values = explainer.shap_values(X_test_sample)
    shap_values_to_use = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

    mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=1)

    shap_importance_list = list(zip(feature_names, mean_abs_shap))
    shap_importance_list.sort(key=lambda x: x[1], reverse=True)

    return shap_importance_list


def compute_shap_importance_nb(model, X_train, X_test, feature_names, background_size=200, n_test=200):
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    # --- background set ---
    if background_size is None or background_size >= X_train_dense.shape[0]:
        background = X_train_dense.astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        idx_bg = rng.choice(X_train_dense.shape[0], background_size, replace=False)
        background = X_train_dense[idx_bg].astype(np.float32)

    explainer = shap.KernelExplainer(model.predict_proba, background)

    # --- test sample ---
    if n_test is None or n_test >= X_test_dense.shape[0]:
        X_test_sample = X_test_dense
    else:
        rng = np.random.default_rng(42)
        idx_test = rng.choice(X_test_dense.shape[0], n_test, replace=False)
        X_test_sample = X_test_dense[idx_test]

    shap_values = explainer.shap_values(X_test_sample, nsamples=300)
    shap_vals = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
    shap_vals = np.array(shap_vals)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_importance_list = list(zip(feature_names, mean_abs_shap.tolist()))
    shap_importance_list.sort(key=lambda x: x[1], reverse=True)

    return shap_importance_list


def find_k_for_target_accuracy(model_template, train_texts, test_texts, y_train, y_test, top_words, accuracy_threshold=0.8):
    vectorizer = TfidfVectorizer(vocabulary=top_words)
    X_train_full = vectorizer.fit_transform(train_texts).toarray()
    X_test_full = vectorizer.transform(test_texts).toarray()

    left, right = 1, len(top_words)
    best_k = None

    model_params = model_template.get_params()

    while left <= right:
        mid = (left + right) // 2
        X_train_k = X_train_full[:, :mid]
        X_test_k = X_test_full[:, :mid]

        model = model_template.__class__(**model_params)
        model.fit(X_train_k, y_train)
        y_pred = model.predict(X_test_k)
        acc = accuracy_score(y_test, y_pred)

        print(f"K={mid}, Accuracy={acc:.3f}")

        if acc >= accuracy_threshold:
            best_k = mid
            right = mid - 1
        else:
            left = mid + 1

    return best_k


def load_glove_embeddings(filepath, embedding_dim=100):
    embeddings_index = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    print(f"Ucitano {len(embeddings_index)} embeddinga iz {filepath}")
    return embeddings_index


def create_embedding_matrix(tokenizer, embeddings_index, max_words=5000, embedding_dim=100):
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    print(f"Kreirana embedding matrica: {embedding_matrix.shape}")
    return embedding_matrix


def prepare_bilstm_data(df, text_column='text', max_words=5000, max_len=200):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df[text_column])
    sequences = tokenizer.texts_to_sequences(df[text_column])
    X = pad_sequences(sequences, maxlen=max_len)
    y = df['spam'].values
    return X, y, tokenizer

def build_bilstm_model(max_words=5000, max_len=200, embedding_dim=100, embedding_matrix=None):
    model = Sequential()
    if embedding_matrix is not None:
        model.add(Embedding(
            input_dim=max_words,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=True #true za fine tuning
        ))
    else:
        model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate_bilstm(df, n_splits=10, epochs=7, batch_size=32, 
                    max_words=5000, max_len=200, 
                    embedding_dim=100, glove_path=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores, precisions, recalls, k_words, top_words_all = [], [], [], [], [], []

    embeddings_index = None
    if glove_path:
        embeddings_index = load_glove_embeddings(glove_path, embedding_dim)

    embedding_matrix = None

    for fold, (train_index, test_index) in enumerate(skf.split(df['text'], df['spam']), 1):
        print(f"\n--- Fold {fold} ---")
        train_texts = df.iloc[train_index]['text']
        test_texts = df.iloc[test_index]['text']
        y_train = df.iloc[train_index]['spam'].values
        y_test = df.iloc[test_index]['spam'].values

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(train_texts)

        X_train_seq = tokenizer.texts_to_sequences(train_texts)
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

        X_test_seq = tokenizer.texts_to_sequences(test_texts)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

        if embeddings_index is not None and embedding_matrix is None:
            embedding_matrix = create_embedding_matrix(
                tokenizer, embeddings_index, max_words=max_words, embedding_dim=embedding_dim
            )

        model = build_bilstm_model(
        max_words=max_words,
        max_len=max_len,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix
    )

        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        model.fit(
            X_train_pad, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        y_pred_prob = model.predict(X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        try:
            np.random.seed(42)

            
            background = shap.kmeans(X_train_pad.astype('float32'), 100)

            
            test_sample_indices = np.random.choice(len(X_test_pad), size=100, replace=False)
            test_sample = X_test_pad[test_sample_indices].astype('float32')

            
            explainer = shap.KernelExplainer(lambda x: model.predict(x).flatten(), background)
            shap_values = explainer.shap_values(test_sample, nsamples=200)

            
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_matrix = shap_values[1]   
                else:
                    shap_matrix = np.mean(shap_values, axis=0)
            else:
                shap_matrix = shap_values

            
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
        top_words_all.append(top_words[:20])

        train_texts = [df['text'].iloc[i] for i in train_index]
        test_texts = [df['text'].iloc[i] for i in test_index]

        k = find_k_for_bilstm_binary_search(train_texts, test_texts, y_train, y_test, 
                                    top_words, accuracy_threshold=0.8,
                                    max_len=max_len,
                                    embedding_matrix=embedding_matrix,
                                    embedding_dim=embedding_dim)
        print(f"[Fold {fold}] K = {k} reci je dovoljno za 80% tacnosti")
        k_words.append(k)

        print(f"Rezultati (Fold {fold}): Accuracy={acc:.3f}, F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    print("\n=== Prosecne metrike za svih 10 foldova (BiLSTM) ===")
    print(f"Accuracy:  {np.mean(accuracies):.3f}")
    print(f"F1 score:  {np.mean(f1_scores):.3f}")
    print(f"Precision: {np.mean(precisions):.3f}")
    print(f"Recall:    {np.mean(recalls):.3f}")

    return f1_scores, k_words, top_words_all

def find_k_for_bilstm_binary_search(train_texts, test_texts, y_train, y_test, 
                                    top_words, accuracy_threshold=0.8, 
                                    max_len=200, embedding_matrix=None, embedding_dim=100):
    left, right = 30, 200 #len(top_words)
    best_k = None

    while left <= right:
        mid = (left + right) // 2
        selected_words = top_words[:mid]
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

        embedding_matrix_k = embedding_matrix[:mid+1, :]
        model = build_bilstm_model(
            max_words=mid+1,
            max_len=max_len,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix_k
        )
        model.fit(X_train_seq, y_train, epochs=3, batch_size=64, verbose=1)

        y_pred_prob = model.predict(X_test_seq)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        print(f"K={mid}, Accuracy={acc:.3f}")

        if acc >= accuracy_threshold:
            best_k = mid
            right = mid - 1
        else:
            left = mid + 1

    return best_k


def main():
    df_downsampled = load_and_balance_data('dataset70k_cleaned_enron.csv', downsample=True)
    df_downsampled['text'] = df_downsampled['text'].apply(clean_text)

    counts = df_downsampled["spam"].value_counts()
    print(f"Broj 1 (spam): {counts.get(1, 0)}")
    print(f"Broj 0 (ham): {counts.get(0, 0)}")

    results = []
    
    print("\n=== Bi-LSTM ===")

    f1_bilstm, k_bilstm, top_words_bilstm = evaluate_bilstm(
    df_downsampled,
    n_splits=10,
    epochs=20,
    batch_size=32,
    max_words=5000,
    max_len=200,
    embedding_dim=100,
    glove_path="glove.6B.100d.txt"
)

    results.append(('BiLSTM', f1_bilstm, k_bilstm, top_words_bilstm))


    print("\n--- Random Forest ---")
    f1_rf, k_rf, top_words_rf = evaluate_model(RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, oob_score=True, n_jobs=-1, random_state=42), df_downsampled, n_splits=10)
    results.append(('Random Forest', f1_rf, k_rf, top_words_rf))

    print("\n--- Decision Tree ---")
    f1_dt, k_dt, top_words_dt = evaluate_model(DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20, min_samples_split=5, min_samples_leaf=2, min_impurity_decrease=0.001, ccp_alpha=0.001, random_state=42), df_downsampled, n_splits=10)
    results.append(('Decision Tree', f1_dt, k_dt, top_words_dt))
    
    print("\n--- Logistic Regression ---")
    f1_lr, k_lr, top_words_lr = evaluate_model(LogisticRegression(penalty='l2', C=100, solver='saga', max_iter=1000, class_weight=None, multi_class='ovr', random_state=42, n_jobs=-1), df_downsampled, n_splits=10)
    results.append(('Logistic Regression', f1_lr, k_lr, top_words_lr))

    print("\n--- Naive Bayes ---")
    f1_nb, k_nb, top_words_nb = evaluate_model(MultinomialNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None), df_downsampled, n_splits=10)
    results.append(('Naive Bayes', f1_nb, k_nb, top_words_nb))

    summary_data = []
    for model_name, f1_list, k_list, top_words_all in results:
        for fold, (f1, k, top_words) in enumerate(zip(f1_list, k_list, top_words_all), 1):
            summary_data.append({
                'Model': model_name,
                'Fold': fold,
                'F1 Score': round(f1, 4),
                'K Words': k,
                'Top 20 Words': ", ".join(top_words)
            })


    summary_df = pd.DataFrame(summary_data)
    print("\n=== Rezime svih modela ===")
    print(summary_df)

    summary_df.to_csv("results_summary.csv", index=False)
    print("\nRezultati su sacuvani u 'results_summary.csv'.")


if __name__ == "__main__":
    main()