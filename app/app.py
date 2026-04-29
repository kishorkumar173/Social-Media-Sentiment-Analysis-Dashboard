import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #00C9A7;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚀 Social Media Sentiment Analysis Dashboard")
st.markdown("### AI-powered insights from customer feedback")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧠 Analyze Text", "📂 Upload CSV"])

# ================= HOME =================
if page == "🏠 Home":
    st.markdown("## 📊 Project Overview")
    st.write("""
    This dashboard analyzes social media comments and classifies them into:
    - Positive 😊
    - Negative 😡
    - Neutral 😐
    """)

    st.info("Use the sidebar to explore features")

# ================= TEXT ANALYSIS =================
elif page == "🧠 Analyze Text":
    st.subheader("Enter Text for Sentiment Analysis")

    text = st.text_area("Type your text here")

    if st.button("Analyze"):
        if text.strip() != "":
            
            # 🔥 Loading Animation
            with st.spinner("Analyzing sentiment..."):
                vec = vectorizer.transform([text])
                pred = model.predict(vec)[0]

            # Output
            if pred == "positive":
                st.success("😊 Positive Sentiment")
            elif pred == "negative":
                st.error("😡 Negative Sentiment")
            else:
                st.warning("😐 Neutral Sentiment")

        else:
            st.warning("Please enter text")

# ================= CSV ANALYSIS =================
elif page == "📂 Upload CSV":
    st.subheader("Upload CSV for Bulk Analysis")

    file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if 'text' not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            # 🔥 Loading Animation
            with st.spinner("Processing data..."):
                df['prediction'] = model.predict(vectorizer.transform(df['text']))

            # ---------------- KPI METRICS ----------------
            st.markdown("### 📊 Key Insights")
            col1, col2, col3 = st.columns(3)

            col1.metric("Positive", (df['prediction'] == "positive").sum())
            col2.metric("Negative", (df['prediction'] == "negative").sum())
            col3.metric("Neutral", (df['prediction'] == "neutral").sum())

            # ---------------- CHARTS ----------------
            st.markdown("### 📈 Sentiment Distribution")

            pie = px.pie(df, names='prediction', title='Sentiment Distribution')
            st.plotly_chart(pie, use_container_width=True)

            bar = px.bar(df['prediction'].value_counts(),
                         title="Sentiment Count",
                         labels={'value': 'Count', 'index': 'Sentiment'})
            st.plotly_chart(bar, use_container_width=True)

            # ---------------- DATA PREVIEW ----------------
            st.markdown("### 📄 Data Preview")
            st.dataframe(df.head())

            # ---------------- DOWNLOAD BUTTON ----------------
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="sentiment_results.pdf",
                mime="application/pdf"
            )