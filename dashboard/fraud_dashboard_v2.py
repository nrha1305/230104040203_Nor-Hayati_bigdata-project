import streamlit as st
import pandas as pd

st.title("🚨 Real-Time Fraud Detection Dashboard")

df = pd.read_parquet("stream_data/realtime_output/")

st.metric("Total Transaksi", len(df))

# 🔥 AMANKAN ERROR DI SINI
if "status" in df.columns:
    st.metric("Total Fraud", len(df[df["status"] == "FRAUD"]))
    st.bar_chart(df["status"].value_counts())
else:
    st.metric("Total Fraud", 0)
    st.warning("⚠️ Kolom 'status' belum ada (Spark belum memproses data)")

st.dataframe(df.tail(10))