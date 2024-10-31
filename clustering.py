import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App
def main():
    st.title("Clustering Techniques on Uploaded Dataset")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview", df.head())
        
        # Check for missing values
        if df.isnull().values.any():
            st.warning("The dataset contains missing values. They will be imputed.")
        
        # Identify numerical and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Sidebar - Clustering algorithm selection
        clustering_algorithm = st.sidebar.selectbox(
            "Select Clustering Algorithm",
            ("K-Means", "Hierarchical Clustering", "DBSCAN")
        )

        # Sidebar - Number of clusters (if applicable)
        if clustering_algorithm != "DBSCAN":
            num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=min(10, len(df)//2), value=3)
        else:
            eps = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
            min_samples = st.sidebar.slider("Minimum Samples", min_value=1, max_value=10, value=5)

        # Sidebar - Feature selection for visualization
        if numeric_cols:
            feature_x = st.sidebar.selectbox("Feature for X-axis", numeric_cols)
            feature_y = st.sidebar.selectbox("Feature for Y-axis", numeric_cols)
        else:
            st.error("No numeric columns available for clustering.")
            return
        
        # Preprocess the data: impute, one-hot encode categorical features and scale numerical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Perform Clustering
        if clustering_algorithm == "K-Means":
            model = KMeans(n_clusters=num_clusters, random_state=0)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clusterer', model)])
            clusters = pipeline.fit_predict(df)
        elif clustering_algorithm == "Hierarchical Clustering":
            model = AgglomerativeClustering(n_clusters=num_clusters)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clusterer', model)])
            clusters = pipeline.fit_predict(df)
        elif clustering_algorithm == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
            scaled_data = preprocessor.fit_transform(df)
            clusters = model.fit_predict(scaled_data)
        
        # Add clusters to the DataFrame
        df['Cluster'] = clusters
        
        # Plot clusters
        st.write("### Clustering Results")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=feature_x, y=feature_y, hue="Cluster", data=df, palette="viridis", ax=ax
        )
        plt.title(f"{clustering_algorithm} Clustering")
        st.pyplot(fig)

        # Display Cluster Counts
        st.write("### Cluster Counts")
        cluster_counts = df['Cluster'].value_counts()
        st.write(cluster_counts)

        # Bar plot for cluster counts
        plt.figure()
        cluster_counts.plot(kind='bar', color='skyblue')
        plt.title("Cluster Counts")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Samples")
        st.pyplot(plt)

if __name__ == "__main__":
    main()
