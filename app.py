import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
from transformers import pipeline
import plotly.express as px
from datetime import datetime
import dateparser
import sqlite3

# ---------------- CONFIG ----------------
st.set_page_config(page_title="VBI Terminal PRO MAX", layout="wide")

import os
SERP_API_KEY = os.getenv("SERPAPI_KEY")
DB_PATH = "vbi_history.db"

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals
                 (time TEXT, source TEXT, headline TEXT, sentiment TEXT, risk TEXT, risk_score REAL)''')
    conn.commit()
    conn.close()

init_db()

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return sentiment, classifier

sentiment_model, classifier = load_models()

# ---------------- DATA FETCH ----------------
def fetch_data(query, region, depth, mode):
    params = {
        "engine": "google",
        "q": query,
        "gl": region.lower(),
        "hl": "en",
        "api_key": SERP_API_KEY
    }

    if mode == "Corporate News":
        params["tbm"] = "nws"
    else:
        params["q"] += " scandal OR complaint OR review OR reddit OR twitter"

    search = GoogleSearch(params)
    results = search.get_dict()

    data = []
    for item in results.get("news_results", [])[:depth]:
        parsed_date = dateparser.parse(item.get("date")) if item.get("date") else datetime.now()

        data.append({
            "Time": parsed_date,
            "Source": item.get("source", "Unknown"),
            "Headline": item.get("title"),
            "Link": item.get("link")
        })

    return data

# ---------------- ANALYSIS ----------------
def analyze(data):
    processed = []

    risk_labels = [
        "Legal Risk",
        "Financial Risk",
        "Technical Failure",
        "PR Crisis",
        "Market Growth"
    ]

    for item in data:
        text = item["Headline"]

        sent = sentiment_model(text)[0]
        label = sent['label']

        if label == "LABEL_0":
            sentiment = "Negative"
        elif label == "LABEL_2":
            sentiment = "Positive"
        else:
            sentiment = "Neutral"

        risk = classifier(text, candidate_labels=risk_labels)

        processed.append({
            "Time": item["Time"],
            "Source": item["Source"],
            "Headline": text,
            "Sentiment": sentiment,
            "Risk": risk['labels'][0],
            "RiskScore": risk['scores'][0]
        })

    return pd.DataFrame(processed)

# ---------------- SAVE HISTORY ----------------
def save_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df_to_save = df.copy()
    df_to_save['Time'] = df_to_save['Time'].astype(str)
    df_to_save.to_sql("signals", conn, if_exists='append', index=False)
    conn.close()

# ---------------- LOAD HISTORY ----------------
def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM signals", conn)
    conn.close()
    if not df.empty:
        df['Time'] = pd.to_datetime(df['Time'])
    return df

# ---------------- UI ----------------
st.title("🛡️ VBI Terminal PRO MAX")

query = st.text_input("Target (Company / Brand)")
region = st.selectbox("Region", ["KZ", "US", "GB", "RU"])
mode = st.radio("Mode", ["Corporate News", "Social Buzz"])
depth = st.slider("Depth", 10, 50, 20)

if st.button("🚀 SCAN"):
    if not query:
        st.warning("Enter target")
    else:
        with st.spinner("Running OSINT scan..."):
            raw = fetch_data(query, region, depth, mode)

            if not raw:
                st.error("No data found")
            else:
                df = analyze(raw)
                df = df.sort_values(by="Time", ascending=False)

                save_to_db(df)

                # METRICS
                pos = len(df[df.Sentiment == "Positive"])
                neg = len(df[df.Sentiment == "Negative"])
                total = len(df)

                rep = 50
                if total > 0:
                    rep = 50 + ((pos - neg) / total) * 50

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Reputation", f"{round(rep,1)}%")
                c2.metric("Signals", total)
                c3.metric("Positive", pos)
                c4.metric("Negative", neg)

                # ALERT
                if total > 0 and (neg / total) > 0.4:
                    st.error("🚨 ALERT: Reputation Risk Spike")

                # CHARTS
                st.subheader("Sentiment Distribution")
                st.plotly_chart(px.pie(df, names="Sentiment"), use_container_width=True)

                st.subheader("Risk Breakdown")
                st.plotly_chart(px.histogram(df, y="Risk"), use_container_width=True)

                st.subheader("Timeline")
                st.plotly_chart(px.scatter(df, x="Time", y="Sentiment", color="Risk", hover_data=["Headline"]), use_container_width=True)

                # DATA
                st.subheader("Live Feed")
                st.dataframe(df, use_container_width=True)

# ---------------- HISTORY ----------------
st.divider()
st.subheader("📊 Historical Trends")

history_df = load_history()

if not history_df.empty:
    history_df['date'] = history_df['Time'].dt.date
    trend = history_df.groupby('date').size().reset_index(name='signals')

    st.plotly_chart(px.line(trend, x='date', y='signals', title="Signal Volume Over Time"), use_container_width=True)
else:
    st.info("No historical data yet")

# ---------------- DOWNLOAD ----------------
if not history_df.empty:
    csv = history_df.to_csv(index=False).encode()
    st.download_button("Download Full History", csv, "vbi_full_history.csv", "text/csv")
