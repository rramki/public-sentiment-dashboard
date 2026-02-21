import streamlit as st
import pandas as pd
import json
import plotly.express as px

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("📊 Public Sentiment Monitoring Dashboard")

# ----------------------------
# LOAD DATA
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV with public posts/comments", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ----------------------------
    # LLM SETUP (Ollama Example)
    # ----------------------------
    llm = Ollama(model="llama3")

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze the sentiment of the following text.

        Return output in JSON format:
        {{
            "sentiment": "Positive/Negative/Neutral",
            "confidence": 0-100,
            "summary": "short explanation"
        }}

        Text: {text}
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    sentiments = []
    confidences = []
    summaries = []

    # ----------------------------
    # SENTIMENT ANALYSIS
    # ----------------------------
    with st.spinner("Analyzing sentiments..."):
        for text in df["text"]:
            response = chain.run(text=text)

            try:
                result = json.loads(response)
                sentiments.append(result["sentiment"])
                confidences.append(result["confidence"])
                summaries.append(result["summary"])
            except:
                sentiments.append("Error")
                confidences.append(0)
                summaries.append("Parsing Error")

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    df["Summary"] = summaries

    # ----------------------------
    # DASHBOARD METRICS
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Posts", len(df))
    col2.metric("Positive", (df["Sentiment"] == "Positive").sum())
    col3.metric("Negative", (df["Sentiment"] == "Negative").sum())

    # ----------------------------
    # SENTIMENT DISTRIBUTION
    # ----------------------------
    st.subheader("Sentiment Distribution")

    fig = px.pie(df, names="Sentiment", title="Sentiment Breakdown")
    st.plotly_chart(fig)

    # ----------------------------
    # DETAILED TABLE
    # ----------------------------
    st.subheader("Detailed Analysis")
    st.dataframe(df)
