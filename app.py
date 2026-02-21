import streamlit as st
import pandas as pd
import json
import plotly.express as px

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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
