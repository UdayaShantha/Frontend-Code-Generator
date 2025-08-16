import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    Load website data from JSON file

    Args:
        file_path (str): Path to the JSON data file

    Returns:
        pd.DataFrame: DataFrame containing the website data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} website records")

        # Display data preview
        print("\nData preview:")
        print(df.head())

        # Check for missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())

        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def engineer_features(df):
    """
    Perform feature engineering on the website data

    Args:
        df (pd.DataFrame): Input DataFrame with raw website data

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    logger.info("Performing feature engineering...")

    # Create a combined text field for text analysis
    text_columns = ['Name', 'Type', 'css-code', 'js-code']
    df['combined_text'] = df[text_columns].apply(lambda row: ' '.join(str(x) for x in row), axis=1)
    logger.info(f"Created combined text from columns: {text_columns}")

    return df


def extract_features(df):
    """
    Extract numerical and text features from the website data

    Args:
        df (pd.DataFrame): DataFrame with engineered features

    Returns:
        pd.DataFrame: DataFrame with extracted features ready for clustering
    """
    logger.info("Extracting features...")

    # Text features using TF-IDF
    logger.info("Vectorizing text features...")
    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    text_features = tfidf.fit_transform(df['combined_text'])
    text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
    text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names)

    # Extract numerical features
    numerical_features = pd.DataFrame()

    # Code length features
    numerical_features['css_length'] = df['css-code'].apply(lambda x: len(str(x)))
    numerical_features['js_length'] = df['js-code'].apply(lambda x: len(str(x)))
    numerical_features['name_length'] = df['Name'].apply(len)

    # Website type features (one-hot encoding)
    website_types = ['food', 'technology', 'news', 'travel', 'e-commerce',
                     'education', 'health', 'social_media', 'finance', 'entertainment']

    for website_type in website_types:
        numerical_features[f'type_{website_type}'] = df['Type'].apply(
            lambda x: 1 if website_type in str(x).lower() else 0
        )

    # Count number of images
    numerical_features['image_count'] = df['Images_and_icons'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    logger.info(f"Created numerical features: {numerical_features.columns.tolist()}")

    # Scale numerical features
    logger.info("Scaling numerical features...")
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features.columns)

    # Combine features
    features_df = pd.concat([text_df, numerical_df], axis=1)
    logger.info(f"Combining features: {text_df.shape[1]} text features and {numerical_df.shape[1]} numerical features")

    return features_df, tfidf, scaler


def apply_pca(features_df, n_components=50):
    """
    Apply PCA to reduce dimensionality of features

    Args:
        features_df (pd.DataFrame): DataFrame with extracted features
        n_components (int): Number of PCA components to keep

    Returns:
        tuple: (pd.DataFrame with PCA features, PCA model)
    """
    logger.info("Applying PCA to reduce dimensionality...")

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(features_df)

    # Create DataFrame with PCA features
    pca_columns = [f'pca_{i}' for i in range(pca_result.shape[1])]
    pca_df = pd.DataFrame(pca_result, columns=pca_columns)

    logger.info(f"Reduced to {pca_result.shape[1]} features with PCA")

    # Explained variance ratio
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"Explained variance: {explained_variance:.2f}%")

    return pca_df, pca


def evaluate_kmeans(features_df):
    """
    Evaluate K-Means clustering with different parameters

    Args:
        features_df (pd.DataFrame): DataFrame with features

    Returns:
        tuple: (best model, best parameters, best score, results DataFrame)
    """
    logger.info("Evaluating K-Means clustering...")

    results = []
    best_score = -1
    best_model = None
    best_params = None

    # Try different number of clusters
    for n_clusters in [2, 3, 4, 5, 6, 7, 8, 10]:
        # Try different initializations
        for init in ['k-means++', 'random']:
            try:
                params = {
                    'n_clusters': n_clusters,
                    'init': init,
                    'n_init': 10,
                    'random_state': 42
                }

                model = KMeans(**params)
                clusters = model.fit_predict(features_df)

                # Calculate silhouette score
                if len(np.unique(clusters)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(features_df, clusters)

                    print(f"  KMeans with {n_clusters} clusters: silhouette score = {score:.4f}")

                    results.append({
                        'algorithm': 'KMeans',
                        'parameters': params,
                        'silhouette_score': score,
                        'n_clusters': len(np.unique(clusters))
                    })

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params

            except Exception as e:
                logger.error(f"Error evaluating KMeans with {n_clusters} clusters: {str(e)}")

    return best_model, best_params, best_score, results


def evaluate_dbscan(features_df):
    """
    Evaluate DBSCAN clustering with different parameters

    Args:
        features_df (pd.DataFrame): DataFrame with features

    Returns:
        tuple: (best model, best parameters, best score, results DataFrame)
    """
    logger.info("Evaluating DBSCAN clustering...")

    results = []
    best_score = -1
    best_model = None
    best_params = None

    # Try different epsilon values and min_samples
    for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for min_samples in [5, 10, 15, 20]:
            try:
                params = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'metric': 'euclidean'
                }

                model = DBSCAN(**params)
                clusters = model.fit_predict(features_df)

                # Calculate silhouette score if there's more than one cluster and no noise points
                if len(np.unique(clusters)) > 1 and -1 not in clusters:
                    score = silhouette_score(features_df, clusters)

                    print(f"  DBSCAN with eps={eps}, min_samples={min_samples}: silhouette score = {score:.4f}")

                    results.append({
                        'algorithm': 'DBSCAN',
                        'parameters': params,
                        'silhouette_score': score,
                        'n_clusters': len(np.unique(clusters))
                    })

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                else:
                    n_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
                    n_noise = np.sum(clusters == -1)
                    print(
                        f"  DBSCAN with eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points")

            except Exception as e:
                logger.error(f"Error evaluating DBSCAN with eps={eps}, min_samples={min_samples}: {str(e)}")

    return best_model, best_params, best_score, results


def evaluate_agglomerative(features_df):
    """
    Evaluate Agglomerative clustering with different parameters

    Args:
        features_df (pd.DataFrame): DataFrame with features

    Returns:
        tuple: (best model, best parameters, best score, results DataFrame)
    """
    logger.info("Evaluating Agglomerative clustering...")

    results = []
    best_score = -1
    best_model = None
    best_params = None

    # Try different parameters
    for n_clusters in [2, 3, 4, 5, 6, 7, 8]:
        for linkage in ['ward', 'complete', 'average']:
            # Skip invalid combinations - ward only works with euclidean
            if linkage == 'ward':
                affinity_options = ['euclidean']
            else:
                affinity_options = ['euclidean', 'manhattan', 'cosine']

            for affinity in affinity_options:
                try:
                    # Fixed: Remove 'affinity' parameter when linkage is 'ward'
                    params = {
                        'n_clusters': n_clusters,
                        'linkage': linkage
                    }

                    # Only add affinity parameter if linkage is not 'ward'
                    if linkage != 'ward':
                        params['affinity'] = affinity

                    model = AgglomerativeClustering(**params)
                    clusters = model.fit_predict(features_df)

                    # Calculate silhouette score
                    if len(np.unique(clusters)) > 1:  # Need at least 2 clusters for silhouette
                        score = silhouette_score(features_df, clusters)

                        print(
                            f"  Agglomerative with {n_clusters} clusters, {linkage} linkage: silhouette score = {score:.4f}")

                        results.append({
                            'algorithm': 'AgglomerativeClustering',
                            'parameters': params,
                            'silhouette_score': score,
                            'n_clusters': len(np.unique(clusters))
                        })

                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_params = params

                except Exception as e:
                    logger.error(f"Error evaluating AgglomerativeClustering: {str(e)}")

    return best_model, best_params, best_score, results


def evaluate_clustering_algorithms(features_df):
    """
    Evaluate different clustering algorithms and return the best one

    Args:
        features_df (pd.DataFrame): DataFrame with features

    Returns:
        tuple: (best model, best algorithm, best parameters, best score, results DataFrame)
    """
    logger.info("Evaluating different clustering algorithms...")

    # Dictionary to store results from each algorithm
    all_results = []

    # Evaluate KMeans
    kmeans_model, kmeans_params, kmeans_score, kmeans_results = evaluate_kmeans(features_df)
    all_results.extend(kmeans_results)

    # Evaluate DBSCAN
    dbscan_model, dbscan_params, dbscan_score, dbscan_results = evaluate_dbscan(features_df)
    all_results.extend(dbscan_results)

    # Evaluate Agglomerative Clustering
    agglom_model, agglom_params, agglom_score, agglom_results = evaluate_agglomerative(features_df)
    all_results.extend(agglom_results)

    # Find the best model overall
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
        best_score = best_result['silhouette_score']
        best_algorithm = best_result['algorithm']
        best_params = best_result['parameters']

        # Get the corresponding model
        if best_algorithm == 'KMeans':
            best_model = kmeans_model
        elif best_algorithm == 'DBSCAN':
            best_model = dbscan_model
        else:  # AgglomerativeClustering
            best_model = agglom_model

        logger.info(f"Best clustering algorithm: {best_algorithm} with score {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_model, best_algorithm, best_params, best_score, results_df
    else:
        logger.warning("No valid clustering results found")
        return None, None, None, -1, pd.DataFrame()


def visualize_clusters(df, features_df, best_model, algorithm_name, output_dir):
    """
    Visualize the clustering results using PCA for dimensionality reduction

    Args:
        df (pd.DataFrame): Original DataFrame with website data
        features_df (pd.DataFrame): DataFrame with features used for clustering
        best_model: Best clustering model
        algorithm_name (str): Name of the best clustering algorithm
        output_dir (str): Directory to save plots
    """
    try:
        logger.info("Visualizing clustering results...")

        # Get cluster assignments
        if hasattr(best_model, 'labels_'):
            clusters = best_model.labels_
        else:
            clusters = best_model.predict(features_df)

        # Add clusters to original dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters

        # Apply PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(features_df)

        # Create a scatter plot of the clusters
        plt.figure(figsize=(12, 8))

        # Use a color palette based on number of clusters
        n_clusters = len(np.unique(clusters))
        if -1 in np.unique(clusters):  # Handle DBSCAN's noise points
            palette = sns.color_palette("tab10", n_clusters)
            palette = ['black'] + list(palette)  # Noise points in black
        else:
            palette = sns.color_palette("tab10", n_clusters)

        scatter = plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=clusters,
            cmap=plt.cm.tab10,
            alpha=0.7,
            s=50
        )

        plt.title(f'Website Clustering using {algorithm_name}', fontsize=15)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)

        # Add legend with cluster numbers
        handles, labels = scatter.legend_elements()
        if -1 in np.unique(clusters):
            legend_labels = ['Noise'] + [f'Cluster {i}' for i in range(n_clusters - 1)]
        else:
            legend_labels = [f'Cluster {i}' for i in range(n_clusters)]

        plt.legend(handles, legend_labels)

        # Add grid and improve style
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'clusters_visualization_{algorithm_name}.png'), dpi=300)
        logger.info(f"Saved cluster visualization to {output_dir}")

        # Sample websites from each cluster
        top_websites_per_cluster = {}
        for cluster_id in np.unique(clusters):
            if cluster_id != -1:  # Skip noise points for DBSCAN
                cluster_websites = df_with_clusters[df_with_clusters['cluster'] == cluster_id]['Name'].tolist()
                top_websites_per_cluster[f"Cluster {cluster_id}"] = cluster_websites[:10]  # Take top 10

                print(f"\nCluster {cluster_id} sample websites:")
                for website in cluster_websites[:10]:
                    print(f"  - {website}")

        # Create a histogram of cluster sizes
        plt.figure(figsize=(10, 6))
        cluster_counts = pd.Series(clusters).value_counts().sort_index()

        # Skip the noise cluster for DBSCAN if it exists
        if -1 in cluster_counts.index:
            noise_count = cluster_counts[-1]
            cluster_counts = cluster_counts[cluster_counts.index != -1]
            plt.bar(range(len(cluster_counts)), cluster_counts.values, alpha=0.8)
            plt.title(f'Cluster Size Distribution ({noise_count} points in noise)', fontsize=15)
        else:
            plt.bar(range(len(cluster_counts)), cluster_counts.values, alpha=0.8)
            plt.title('Cluster Size Distribution', fontsize=15)

        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Websites', fontsize=12)
        plt.xticks(range(len(cluster_counts)), [f'Cluster {i}' for i in cluster_counts.index])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(output_dir, f'cluster_distribution_{algorithm_name}.png'), dpi=300)

        return df_with_clusters

    except Exception as e:
        logger.error(f"Error visualizing clusters: {str(e)}")
        return df


def save_model(model, algorithm_name, params, pipeline_components, output_dir):
    """
    Save the trained model and related components

    Args:
        model: The trained clustering model
        algorithm_name (str): Name of the algorithm
        params (dict): Parameters of the model
        pipeline_components (dict): Dictionary of pipeline components
        output_dir (str): Directory to save the model
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Save the model
        model_path = os.path.join(output_dir, f"{algorithm_name}_model.pkl")
        joblib.dump(model, model_path)

        # Save pipeline components
        for name, component in pipeline_components.items():
            component_path = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(component, component_path)

        # Save model info
        model_info = {
            'algorithm': algorithm_name,
            'parameters': params,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'components': list(pipeline_components.keys())
        }

        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=4)

        logger.info(f"Model and components saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")


def analyze_clusters(df_with_clusters):
    """
    Analyze the characteristics of each cluster

    Args:
        df_with_clusters (pd.DataFrame): DataFrame with cluster assignments
    """
    logger.info("Analyzing cluster characteristics...")

    # Get unique clusters
    unique_clusters = df_with_clusters['cluster'].unique()

    # Analyze each cluster
    for cluster_id in unique_clusters:
        # Skip noise points (DBSCAN)
        if cluster_id == -1:
            continue

        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        cluster_size = len(cluster_df)

        print(f"\nCluster {cluster_id} Analysis ({cluster_size} websites):")

        # Most common website types
        types = []
        for t in cluster_df['Type']:
            types.extend([x.strip() for x in str(t).split(',')])

        type_counts = pd.Series(types).value_counts()
        print("  Most common website types:")
        for website_type, count in type_counts.head(5).items():
            percentage = (count / cluster_size) * 100
            print(f"    - {website_type}: {count} ({percentage:.1f}%)")

        # Average code length
        avg_css_length = cluster_df['css-code'].apply(lambda x: len(str(x))).mean()
        avg_js_length = cluster_df['js-code'].apply(lambda x: len(str(x))).mean()
        print(f"  Average CSS code length: {avg_css_length:.1f} characters")
        print(f"  Average JS code length: {avg_js_length:.1f} characters")

        # Average number of images
        avg_images = cluster_df['Images_and_icons'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).mean()
        print(f"  Average number of images: {avg_images:.1f}")


def main(data_file_path=None, output_dir="output", pca_components=50, random_seed=42):
    """
    Main function to run the website clustering workflow

    Args:
        data_file_path (str): Path to the JSON data file
        output_dir (str): Directory to save output files
        pca_components (int): Number of PCA components to use
        random_seed (int): Random seed for reproducibility
    """
    start_time = time.time()
    logger.info("Starting website clustering model training workflow...")

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Set default data file path if not provided
    if data_file_path is None:
        data_file_path = "website_dataset.json"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data from: {data_file_path}")
    df = load_data(data_file_path)

    # Engineer features
    df = engineer_features(df)

    # Extract features
    features_df, tfidf_vectorizer, scaler = extract_features(df)

    # Apply PCA for dimensionality reduction
    pca_df, pca_model = apply_pca(features_df, n_components=pca_components)

    # Evaluate clustering algorithms
    best_model, best_algorithm, best_params, best_score, results_df = evaluate_clustering_algorithms(pca_df)

    # If we found a good model, visualize and save it
    if best_model is not None:
        # Visualize clustering results
        df_with_clusters = visualize_clusters(df, pca_df, best_model, best_algorithm, output_dir)

        # Analyze clusters
        analyze_clusters(df_with_clusters)

        # Save model and pipeline components
        pipeline_components = {
            'tfidf_vectorizer': tfidf_vectorizer,
            'scaler': scaler,
            'pca': pca_model
        }

        save_model(best_model, best_algorithm, best_params, pipeline_components, output_dir)

        # Save results of all algorithm evaluations
        results_df.to_csv(os.path.join(output_dir, 'clustering_evaluation_results.csv'), index=False)

        # Save clustered data
        df_with_clusters.to_csv(os.path.join(output_dir, 'website_clusters.csv'), index=False)

    elapsed_time = time.time() - start_time
    logger.info(f"Workflow completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train website clustering model')
    parser.add_argument('--data', type=str, help='Path to the JSON data file')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--components', type=int, default=50, help='Number of PCA components')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    try:
        args = parser.parse_args()
        main(data_file_path=args.data, output_dir=args.output, pca_components=args.components, random_seed=args.seed)
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        # Try with default arguments
        logger.info("No arguments provided or error in arguments. Using default values.")
        main()