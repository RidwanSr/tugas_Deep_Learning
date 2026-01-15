
import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ======================================
# LOAD DATASET
# ======================================
def load_data_20221310038():
    url = "https://raw.githubusercontent.com/rasyidev/well-known-datasets/main/juli2train.csv"
    df = pd.read_csv(url)
    return df

# ======================================
# PREPROCESSING TEXT
# ======================================
def clean_text_20221310038(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ======================================
# LABEL ENCODING (Not strictly needed for this dataset's 'label' column)
# ======================================
def label_encoding_20221310038(label):
    return 1 if label == 'positif' else 0

# ======================================
# TRAIN MODEL
# ======================================
def train_model_20221310038(df):
    df['clean_text'] = df['tweet'].apply(clean_text_20221310038)
    # The 'label' column is already numeric (0/1), so no need for re-encoding from 'sentiment' column
    # df['label'] = df['sentiment'].apply(label_encoding_20221310038)

    X_train, _, y_train, _ = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=0.2,
        random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    return model, tfidf

# ======================================
# PREDICTION
# ======================================
def predict_sentiment_20221310038(text, model, tfidf):
    text = clean_text_20221310038(text)
    vector = tfidf.transform([text])
    prediction = model.predict(vector)

    return "Positif" if prediction[0] == 1 else "Negatif"

# ======================================
# STREAMLIT UI
# ======================================
def main_20221310038():
    st.set_page_config(
        page_title="Sentiment Analysis UAS",
        layout="centered"
    )

    st.title("Sentiment Analysis Debat Capres 2024")
    st.write("Metode: Logistic Regression")
    st.write("NPM: 20221310038")

    with st.spinner("Memuat dataset dan melatih model..."):
        df = load_data_20221310038()
        model, tfidf = train_model_20221310038(df)

    user_text = st.text_area(
        "Masukkan teks tanggapan masyarakat:",
        height=150
    )

    if st.button("Prediksi Sentimen"):
        if user_text.strip() == "":
            st.warning("Teks tidak boleh kosong")
        else:
            result = predict_sentiment_20221310038(
                user_text, model, tfidf
            )

            if result == "Positif":
                st.success(f"Hasil Sentimen: {result}")
            else:
                st.error(f"Hasil Sentimen: {result}")

# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    main_20221310038()
