import streamlit as st
import nltk
import stanza
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer

# streamlit run Text_NLP_Processing.py

# Download necessary resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")
stanza.download("en")

# Initialize Stanza and Sentiment Analyzer
nlp_stanza = stanza.Pipeline(lang="en", processors="tokenize,pos")
sia = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("🔍 NLP Processing Web Application")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a text file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("📄 Uploaded File Content")
    st.write(text[:500] + "...")  # Displaying only first 500 characters

    # Tokenization
    tokens = word_tokenize(text)
    tokens_lower = [word.lower() for word in tokens if word.isalpha()]

    # Stopword Removal
    stop_words = set(stopwords.words("english"))
    tokens_no_stopwords = [word for word in tokens_lower if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens_no_stopwords]

    # POS Tagging using Stanza
    doc = nlp_stanza(" ".join(tokens_no_stopwords))
    pos_tags = [(word.text, word.xpos) for sent in doc.sentences for word in sent.words]

    # Keyword Extraction
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:10]  # Top 10 keywords

    # Sentiment Analysis
    sentiment_scores = sia.polarity_scores(text)
    sentiment_label = "Positive 😀" if sentiment_scores["compound"] >= 0.05 else (
        "Negative 😠" if sentiment_scores["compound"] <= -0.05 else "Neutral 😐"
    )

    # Buttons for NLP Tasks
    if st.button("🔤 Show Stemming Results"):
        st.subheader("📝 Stemmed Words")
        st.write(", ".join(stemmed_tokens[:30]) + " ...")  # Display first 30 words

    if st.button("🔑 Extract Keywords"):
        st.subheader("📌 Top Keywords")
        st.write(", ".join(keywords))

    if st.button("😊 Sentiment Analysis"):
        st.subheader("📊 Sentiment Scores")
        st.write(sentiment_scores)
        st.write(f"**Overall Sentiment:** {sentiment_label}")

        # Sentiment Bar Chart
        fig, ax = plt.subplots()
        ax.bar(["Positive", "Neutral", "Negative"], [sentiment_scores["pos"], sentiment_scores["neu"], sentiment_scores["neg"]], color=["green", "blue", "red"])
        ax.set_ylabel("Score")
        ax.set_title("Sentiment Analysis")
        st.pyplot(fig)

    if st.button("☁ Generate Word Cloud"):
        st.subheader("🌥 Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
