import streamlit as st
import pandas as pd
from GoogleNews import GoogleNews
from transformers import pipeline
import plotly.express as px
import dateparser
from datetime import datetime
from utils import save_to_db, load_from_db, format_display_time

st.set_page_config(
    page_title="VBI Terminal: Kazakhstan", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# SerpAPI Key
# ===========================
SERP_API_KEY = st.secrets.get("SERPAPI_KEY")
if not SERP_API_KEY:
    st.error("🚨 SERPAPI_KEY not found. Add it in Secrets.")
    st.stop()

# ===========================
# Model
# ===========================
@st.cache_resource
def load_neural_engine():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
analyzer = load_neural_engine()

# ===========================
# Fetch Intelligence
# ===========================
def fetch_intelligence(query, region, depth, mode):
    lang_set = 'en'
    gn = GoogleNews(lang=lang_set, region=region)
    gn.clear()
    
    if mode == "Social Buzz (Risk)":
        final_query = f'{query} scandal OR opinion OR review OR complaint OR leak OR reddit OR twitter'
    else:
        final_query = query

    try:
        gn.search(final_query)
        results = gn.result()
        if results and len(results) < depth:
            try:
                gn.getpage(2)
                results = gn.result()
            except Exception:
                pass
    except Exception:
        return []

    clean_data = []
    seen_titles = set()
    for item in results[:depth]:
        title = item.get('title', '')
        if title in seen_titles: continue
        seen_titles.add(title)
        raw_date = item.get('date', '')
        parsed_date = dateparser.parse(raw_date) if raw_date else datetime.now()
        source_label = item.get('media', 'Unknown Node')
        if any(x in title.lower() for x in ['twitter', 'reddit', 'post', 'blog', 'users']):
            source_label = "Social/Forum"
        clean_data.append({
            "Timestamp": parsed_date,
            "Source": source_label,
            "Headline": title,
            "Link": item.get('link', '#')
        })
    return clean_data

# ===========================
# Sidebar
# ===========================
with st.sidebar:
    st.header("🛰️ MONITORING CONFIG")
    st.write("Target Region: **Kazakhstan (KZ)**")
    target_region = st.selectbox("Geo-Location Node", ["KZ", "US", "GB", "RU"], index=0)
    st.divider()
    source_mode = st.radio(
        "Select Signal Type:",
        ["Corporate News", "Social Buzz (Risk)"],
        captions=["Official Media & PR", "Opinions, Scandals & Noise"]
    )
    scan_depth = st.slider("Signal Depth (Items)", 10, 50, 30)
    st.divider()
    st.info(f"System Status: **ONLINE**\n\nActive Mode: {source_mode}")

# ===========================
# Main UI
# ===========================
st.title("🛡️ VBI: Real-Time Reputation Monitor")
st.markdown("Automated OSINT & Risk Categorization System")

target_query = st.text_input("TARGET ENTITY (e.g. Air Astana, Kaspi, KMG):", placeholder="Enter brand name...")

if st.button("INITIATE SCAN"):
    if not target_query:
        st.warning("⚠️ Please enter a target name.")
    else:
        with st.spinner(f"📡 Intercepting {source_mode} signals for '{target_query}'..."):
            raw_data = fetch_intelligence(target_query, target_region, scan_depth, source_mode)
            if not raw_data:
                st.error("No signals detected. Try a broader keyword or switch Source Layer.")
            else:
                processed_data = []
                risk_vectors = ["Legal/Compliance", "Financial Risk", "Technical Failure", "Market Expansion", "PR Crisis"]
                sentiment_cats = ["Positive", "Negative", "Neutral"]

                for i, item in enumerate(raw_data):
                    risk_out = analyzer(item['Headline'], candidate_labels=risk_vectors)
                    sent_out = analyzer(item['Headline'], candidate_labels=sentiment_cats)
                    processed_data.append({
                        "Time": item['Timestamp'],
                        "DisplayTime": item['Timestamp'].strftime('%H:%M - %d %b'),
                        "Source": item['Source'],
                        "Headline": item['Headline'],
                        "Risk Category": risk_out['labels'][0],
                        "Risk Score": risk_out['scores'][0],
                        "Sentiment": sent_out['labels'][0]
                    })

                df = pd.DataFrame(processed_data).sort_values(by='Time', ascending=False)

                # Save signals safely
                save_to_db(df)

                # Metrics
                pos_count = len(df[df['Sentiment'] == "Positive"])
                neg_count = len(df[df['Sentiment'] == "Negative"])
                total_signals = len(df)
                net_score = (pos_count - neg_count) / total_signals if total_signals > 0 else 0
                rep_index = 50 + (net_score * 50)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Reputation Index", f"{round(rep_index,1)}%", delta=f"{source_mode} Mode")
                c2.metric("Total Signals", total_signals)
                c3.metric("Positive Coverage", pos_count)
                c4.metric("Active Threats", neg_count, delta_color="inverse")

                # Charts
                g1, g2 = st.columns([1,1])
                with g1:
                    fig_pie = px.pie(df, names='Sentiment', hole=0.5, title="Sentiment Share",
                                     color='Sentiment',
                                     color_discrete_map={"Positive":"#10b981","Negative":"#ef4444","Neutral":"#64748b"})
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with g2:
                    risk_counts = df['Risk Category'].value_counts().reset_index()
                    risk_counts.columns = ['Risk', 'Count']
                    fig_bar = px.bar(risk_counts, x='Count', y='Risk', orientation='h', title="Risk Classification",
                                     color='Risk', color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0")
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("📡 Live Intelligence Log")
                display_df = df[['DisplayTime','Source','Headline','Risk Category','Sentiment']].copy()
                def sentiment_color(val):
                    color = '#10b981' if val=='Positive' else '#ef4444' if val=='Negative' else '#94a3b8'
                    return f'color: {color}; font-weight: bold'
                st.dataframe(display_df.style.map(sentiment_color, subset=['Sentiment']), use_container_width=True, hide_index=True)

                # CSV Download
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
                csv = df.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button("📥 Download Full Audit (CSV)", csv, f"VBI_Report_{target_query}_{timestamp_str}.csv", "text/csv")
