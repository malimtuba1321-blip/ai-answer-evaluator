# app.py
import streamlit as st
import pandas as pd
from evaluator import evaluate

st.set_page_config(page_title="AI Answer Evaluation", page_icon="🤖", layout="wide")

st.title("🤖 Real-Time AI Answer Evaluation Using NLP")
st.caption("Compare an AI-generated answer with a reference answer using BLEU, ROUGE, and BERTScore.")

st.subheader("📝 Inputs")

question = st.text_area("Question", placeholder="Enter the question here...", height=80)

col1, col2 = st.columns(2)
with col1:
    reference = st.text_area("Reference Answer (Expected)", placeholder="Enter the model answer...", height=160)
with col2:
    ai_answer = st.text_area("AI-Generated Answer", placeholder="Enter the AI answer...", height=160)

if st.button("🚀 Evaluate", type="primary", use_container_width=True):
    if not reference.strip() or not ai_answer.strip():
        st.warning("⚠️ Please enter both answers.")
    else:
        with st.spinner("Computing scores... (first run may take ~30s)"):
            results = evaluate(reference, ai_answer)

        st.success("✅ Evaluation complete!")

        st.subheader("📊 Evaluation Scores")
        a, b, c = st.columns(3)
        a.metric("BLEU Score", f"{results['BLEU']:.4f}")
        b.metric("ROUGE-L Score", f"{results['ROUGE-L']:.4f}")
        c.metric("BERTScore F1", f"{results['BERTScore-F1']:.4f}")

        st.subheader("📋 Detailed Metrics")
        df = pd.DataFrame([(k, v) for k, v in results.items()], columns=["Metric", "Score"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("📈 Visual Comparison")
        chart_df = pd.DataFrame({
            "Metric": ["BLEU", "ROUGE-L", "BERTScore-F1", "Composite"],
            "Score": [results["BLEU"], results["ROUGE-L"], results["BERTScore-F1"], results["Composite"]]
        }).set_index("Metric")
        st.bar_chart(chart_df)

        st.subheader("🧠 Interpretation")
        st.markdown(f"""
        - **BLEU = {results['BLEU']:.4f}** → Word/phrase matching
        - **ROUGE-L = {results['ROUGE-L']:.4f}** → Longest common subsequence overlap
        - **BERTScore-F1 = {results['BERTScore-F1']:.4f}** → Semantic similarity
        - **Composite = {results['Composite']:.4f}** → Average of the three

        📌 Score above **0.7** = strong match | below **0.4** = weak match
        """)