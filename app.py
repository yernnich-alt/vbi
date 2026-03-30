import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from datetime import datetime
from utils import save_to_db
import dateparser
import requests
import time

st.set_page_config(
    page_title="VBI Terminal: Kazakhstan",
    page_icon="🛡️",
    layout="wide"
)

# ===========================
# API KEY
# ===========================
SERP_API_KEY = st.secrets.get("SERPAPI_KEY")
if not SERP_API_KEY:
    st.error("🚨 SERPAPI_KEY not found. Add it in Secrets.")
    st.stop()

# ===========================
# AI Model
# ===========================
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

analyzer = load_model()

risk_vectors = ["Legal/Compliance", "Financial Risk", "Technical Failure", "Market Expansion", "PR Crisis"]
sentiment_cats = ["Positive", "Negative", "Neutral"]

# ===========================
# FETCH DATA (SerpAPI)
# ===========================
def fetch_intelligence(query, region, depth, mode, api_key):
    if mode == "Social Buzz (Risk)":
        query = f"{query} scandal OR leak OR complaint OR opinion"

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "hl": "en",
        "gl": region,
        "tbm": "nws",
        "num": depth,
        "api_key": api_key
    }

    for _ in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = data.get("news_results", [])
            clean_data = []

            for item in results:
                title = item.get("title", "")
                if not title:
                    continue

                raw_date = item.get("date", "")
                parsed_date = dateparser.parse(raw_date) if raw_date else datetime.now()

                clean_data.append({
                    "Timestamp": parsed_date,
                    "Source": item.get("source", "Unknown"),
                    "Headline": title,
                    "Link": item.get("link", "#")
                })

            return clean_data

        except Exception as e:
            print("SerpAPI error:", e)
            time.sleep(2)

    return []

# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.header("🛰️ MONITORING CONFIG")

    target_region = st.selectbox("Region", ["KZ", "US", "GB", "RU"], index=0)

    source_mode = st.radio(
        "Mode",
        ["Corporate News", "Social Buzz (Risk)"]
    )

    scan_depth = st.slider("Depth", 10, 50, 20)

# ===========================
# MAIN
# ===========================
st.title("🛡️ VBI OSINT Monitor")

query = st.text_input("Enter company / brand")

if st.button("SCAN"):
    if not query:
        st.warning("Enter query")
        st.stop()

    with st.spinner("Scanning..."):
        raw_data = fetch_intelligence(query, target_region, scan_depth, source_mode, SERP_API_KEY)

        if not raw_data:
            st.error("No data")
            st.stop()

        processed_data = []

        for item in raw_data:
            headline = item.get("Headline", "")
            if not headline:
                continue

            risk_out = analyzer(headline, candidate_labels=risk_vectors)
            sent_out = analyzer(headline, candidate_labels=sentiment_cats)

            processed_data.append({
                "Time": item.get("Timestamp"),
                "Source": item.get("Source"),
                "Headline": headline,
                "Risk": risk_out["labels"][0],
                "Sentiment": sent_out["labels"][0]
            })

        if not processed_data:
            st.warning("No valid data")
            st.stop()

        df = pd.DataFrame(processed_data)

        # SAVE
        save_to_db(df)

        # METRICS
        pos = len(df[df["Sentiment"] == "Positive"])
        neg = len(df[df["Sentiment"] == "Negative"])
        total = len(df)

        score = 50 + ((pos - neg) / total * 50)

        c1, c2, c3 = st.columns(3)
        c1.metric("Reputation", round(score, 1))
        c2.metric("Signals", total)
        c3.metric("Negative", neg)

        # CHART
        fig = px.pie(df, names="Sentiment")
        st.plotly_chart(fig)

        # TABLE
        st.dataframe(df)

        # DOWNLOAD
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv)
