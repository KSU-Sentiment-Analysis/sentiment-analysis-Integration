import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from dataset_handler import load_csv, process_data

def load_and_prepare_data(dataset_name="dataset_3.csv"):
    """
    Btw here is the key on how to read the data
      - 'negative' if rating < 3
      - 'neutral'  if rating == 3
      - 'positive' if rating > 3

    Returns:
        X  Review texts.
        y  Three-class sentiment labels.
    """
    df = load_csv(dataset_name)
    # need to start using logger instead of print statments hahaha
    print("-" * 60)
    print(f"loaded {dataset_name} with encoding=utf-8 and delimiter=','")
    print(f"Columns detected: {df.columns.tolist()}")

    processed_df = process_data(df)
    if processed_df is None or processed_df.empty:
        raise ValueError("No valid data available for training.")

    print("-" * 60)
    print("Processed Data (first 5 rows):")
    print(processed_df.head().to_string(index=False))
    print("-" * 60)

    def map_rating_to_sentiment(r):
        if r > 3:
            return "positive"
        elif r == 3:
            return "neutral"
        else:
            return "negative"

    X = processed_df["review_text"]
    y = processed_df["rating"].apply(map_rating_to_sentiment)
    return X, y

def train_naive_bayes_model(X_train, y_train):
    """
    Creating a pipline for TF-IDF + MultinomialNB and using grid search to get  hyperparameters. kinda works
    """
    print("Starting search here for Naive Bayes hyperparameters...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])

    # i just guessed :)
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_df": [0.9, 1.0],
        "tfidf__min_df": [1, 2],
        "nb__alpha": [0.1, 1.0, 5.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("-" * 60)
    print("=== Grid Search Results ===")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Macro-F1 Score: {grid_search.best_score_:.4f}")
    print("-" * 60)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Making the whole classification report and confusing matrix as the req in kick off slides
    """
    y_pred = model.predict(X_test)
    print("=== Evaluation on Test Set ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix (order: negative, neutral, positive):")
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
    print(cm)
    print("-" * 60)
    return y_pred

def save_model(model, filename="naive_bayes_3class.pkl"):
    """
    Saveing the trained model pipeline to disk.
    """
    joblib.dump(model, filename)
    print(f"✅ Model saved to '{filename}'\n")

def load_custom_model(filename="naive_bayes_3class.pkl"):
    """
    Loading a prev saved custom model from disk.
    """
    model = joblib.load(filename)
    print(f"Custom model loaded from '{filename}'\n")
    return model

def train_and_save_model_from_data(X, y):
    print("\n" + "=" * 60)
    print("TRAINING CUSTOM THREE-CLASS SENTIMENT MODEL (Naive Bayes)")
    print("=" * 60 + "\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_naive_bayes_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    return model

def train_and_save_model(dataset_name="dataset_3.csv"):
    """
    Basically for starting the model from scratch
    """
    X, y = load_and_prepare_data(dataset_name)
    return train_and_save_model_from_data(X, y)
