# 🧩 Customer Segmentation App with Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------
# Title and description
# --------------------------------------
st.title("📊 Customer Segmentation App")
st.write("Upload your marketing dataset and discover customer segments using KMeans clustering.")

# --------------------------------------
# File upload
# --------------------------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # Basic cleaning
    if 'Year_Birth' in df.columns:
        df['Age'] = 2025 - df['Year_Birth']

    # Drop missing values
    df = df.dropna()

    # Select numeric features
    features = df.select_dtypes(include=np.number)
    for col in features.columns:
        if "ID" in col or "Id" in col:
            features = features.drop(columns=[col])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Sidebar controls
    st.sidebar.header("⚙️ Clustering Settings")
    k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 4)

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Silhouette Score
    sil_score = silhouette_score(X_scaled, df['Cluster'])
    st.sidebar.write(f"📈 Silhouette Score: {sil_score:.3f}")

    # Cluster Summary (numeric only)
    st.write("### 📊 Cluster Summary (mean values)")
    numeric_cols = df.select_dtypes(include=np.number).columns
    st.dataframe(df.groupby("Cluster")[numeric_cols].mean())


    # PCA for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=df['Cluster'], palette="Set2", ax=ax)
    plt.title("Customer Clusters (PCA Projection)")
    st.pyplot(fig)

    # Download clustered dataset
    st.write("### 💾 Download Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Clustered Dataset", csv, "clustered_customers.csv", "text/csv")
