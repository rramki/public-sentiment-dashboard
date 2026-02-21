import streamlit as st
import pandas as pd
import json
import plotly.express as px
from PIL import Image
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load logo
logo = Image.open("logo.png")

# Display logo
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=750)

st.title("SASTRA.ai")

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("📊 Public Sentiment Monitoring Dashboard")

uploaded_file = st.file_uploader("data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"]
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze sentiment and return JSON:
        {{
            "sentiment": "Positive/Negative/Neutral",
            "confidence": 0-100
        }}

        Text: {text}
        """
    )

    chain = prompt | llm

    sentiments = []
    confidences = []

    with st.spinner("Analyzing..."):
        for text in df["text"]:
            response = chain.invoke({"text": text})

            try:
                result = json.loads(response.content)
                sentiments.append(result["sentiment"])
                confidences.append(result["confidence"])
            except:
                sentiments.append("Error")
                confidences.append(0)

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences

    st.metric("Total Posts", len(df))

    fig = px.pie(df, names="Sentiment")
    st.plotly_chart(fig)

    st.dataframe(df)


st.title("📊 Social Media Sentiment Monitor")

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("Settings")

company_name = st.sidebar.text_input("Enter Company Name", "Tesla")

news_api_key = st.sidebar.text_input("News API Key", type="password")
reddit_client_id = st.sidebar.text_input("Reddit Client ID")
reddit_client_secret = st.sidebar.text_input("Reddit Client Secret")
reddit_user_agent = st.sidebar.text_input("Reddit User Agent", "sentiment_app")

analyze_button = st.sidebar.button("Analyze")

# ---------------------------
# LOAD SENTIMENT MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

# ---------------------------
# FETCH NEWS DATA
# ---------------------------
def fetch_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={news_api_key}"
    response = requests.get(url)
    data = response.json()
    articles = []

    if "articles" in data:
        for article in data["articles"]:
            if article["title"]:
                articles.append(article["title"])

    return articles

# ---------------------------
# FETCH REDDIT DATA
# ---------------------------
def fetch_reddit(company):
    reddit = praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent
    )

    posts = []
    for submission in reddit.subreddit("all").search(company, limit=20):
        posts.append(submission.title)

    return posts

# ---------------------------
# ANALYSIS SECTION
# ---------------------------
if analyze_button:

    all_text = []

    with st.spinner("Fetching News..."):
        news_data = fetch_news(company_name)
        all_text.extend(news_data)

    with st.spinner("Fetching Reddit Posts..."):
        reddit_data = fetch_reddit(company_name)
        all_text.extend(reddit_data)

    if len(all_text) == 0:
        st.warning("No data found.")
    else:
        st.success(f"Collected {len(all_text)} posts/articles")

        # Sentiment Analysis
        results = sentiment_model(all_text)

        df = pd.DataFrame(results)
        sentiment_counts = df["label"].value_counts()

        # Display Results
        st.subheader("Sentiment Distribution")

        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Sample Data")
        display_df = pd.DataFrame({
            "Text": all_text,
            "Sentiment": df["label"]
        })

        st.dataframe(display_df.head(10))
