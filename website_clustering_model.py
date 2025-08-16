import numpy as np
import pandas as pd
import json
import re
import os
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ParameterGrid
from scipy.sparse import hstack, csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import warnings
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

class WebsiteClusteringModel:
    def __init__(self, data_path, output_dir='model_output', n_components=50, random_state=42):
        """
        Initialize the clustering model.
        
        Parameters:
        -----------
        data_path : str
            Path to the JSON dataset file
        output_dir : str
            Directory to save model outputs
        n_components : int
            Number of components for dimensionality reduction
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.n_components = n_components
        self.random_state = random_state
        self.dataset = None
        self.features = None
        self.best_model = None
        self.best_encoder = None
        self.best_n_clusters = None
        self.best_score = -1
        self.vectorizers = {}
        self.dimensionality_reducers = {}
        self.scalers = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize TensorFlow settings
        self._set_tensorflow_config()
        
    def _set_tensorflow_config(self):
        """Configure TensorFlow settings"""
        # Set memory growth to avoid consuming all GPU memory
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"GPU(s) available: {[device.name for device in physical_devices]}")
            else:
                print("No GPU available, using CPU")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    
    def load_data(self):
        """Load and preprocess the dataset"""
        print(f"Loading data from {self.data_path}...")
        start_time = time.time()
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
            
        print(f"Loaded {len(self.dataset)} website samples in {time.time() - start_time:.2f} seconds")
        
        # Validate data structure
        required_fields = ["Name", "Type", "css-code", "js-code", "Images_and_icons"]
        missing_fields = []
        
        for field in required_fields:
            if not all(field in item for item in self.dataset):
                missing_count = sum(1 for item in self.dataset if field not in item)
                missing_fields.append(f"{field} missing in {missing_count} records")
        
        if missing_fields:
            print("Warning: Data quality issues detected:")
            for issue in missing_fields:
                print(f"- {issue}")
        
        return self

    def _clean_text(self, text):
        """Clean text data by removing extra whitespace, comments, etc."""
        if not isinstance(text, str):
            return ""
        
        # Remove comments
        text = re.sub(r'/\*[\s\S]*?\*/', ' ', text)  # CSS multi-line comments
        text = re.sub(r'//.*', ' ', text)           # JS single-line comments
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _extract_css_features(self, css_text):
        """Extract meaningful features from CSS"""
        if not isinstance(css_text, str):
            return {}
        
        features = {}
        
        # Extract color values
        colors = re.findall(r'#([0-9a-fA-F]{3,6})|rgba?\(\s*(\d+\s*,\s*\d+\s*,\s*\d+\s*,?\s*[\d.]*\s*)\)', css_text)
        features['color_count'] = len(colors)
        
        # Extract font families
        fonts = re.findall(r'font-family\s*:\s*([^;]+)', css_text)
        features['font_count'] = len(fonts)
        
        # Count media queries (responsiveness)
        media_queries = re.findall(r'@media', css_text)
        features['media_query_count'] = len(media_queries)
        
        # Count animations and transitions
        animations = re.findall(r'@keyframes|animation|transition', css_text)
        features['animation_count'] = len(animations)
        
        # Count flex/grid usage (modern layout)
        modern_layout = re.findall(r'display\s*:\s*(flex|grid)', css_text)
        features['modern_layout_count'] = len(modern_layout)
        
        return features

    def _extract_js_features(self, js_text):
        """Extract meaningful features from JavaScript"""
        if not isinstance(js_text, str):
            return {}
        
        features = {}
        
        # Count event listeners
        event_listeners = re.findall(r'addEventListener|on\w+\s*=', js_text)
        features['event_listener_count'] = len(event_listeners)
        
        # Check for modern JS features
        es6_features = re.findall(r'=>|const|let|Promise|async|await|class\s+\w+', js_text)
        features['es6_feature_count'] = len(es6_features)
        
        # Count DOM manipulations
        dom_operations = re.findall(r'getElementById|querySelector|createElement|appendChild', js_text)
        features['dom_operation_count'] = len(dom_operations)
        
        # Check for frameworks
        frameworks = {
            'jquery': re.findall(r'\$\(|\jQuery', js_text),
            'react': re.findall(r'React|render\s*\(|Component|useState|useEffect', js_text),
            'vue': re.findall(r'Vue|v-if|v-for|v-model|v-on', js_text),
            'angular': re.findall(r'ng-|angular|\[\[.+?\]\]', js_text)
        }
        
        for framework, matches in frameworks.items():
            features[f'{framework}_count'] = len(matches)
            
        return features

    def extract_features(self, vectorize=True):
        """Extract features from the dataset"""
        print("Extracting features...")
        start_time = time.time()
        
        # Prepare feature dataframes
        text_features = []
        numerical_features = []
        categorical_features = []
        
        for item in self.dataset:
            # Text features
            css_code = self._clean_text(item.get('css-code', ''))
            js_code = self._clean_text(item.get('js-code', ''))
            site_name = item.get('Name', '')
            site_type = item.get('Type', '')
            
            text_features.append({
                'css_code': css_code,
                'js_code': js_code,
                'site_name': site_name,
                'site_type': site_type
            })
            
            # Numerical features
            css_feats = self._extract_css_features(css_code)
            js_feats = self._extract_js_features(js_code)
            
            num_feats = {
                'css_length': len(css_code),
                'js_length': len(js_code),
                'image_count': len(item.get('Images_and_icons', [])),
                **css_feats,
                **js_feats
            }
            numerical_features.append(num_feats)
            
            # Categorical features
            site_types = [t.strip() for t in site_type.split(',')]
            cat_feats = {
                'site_types': site_types
            }
            categorical_features.append(cat_feats)
        
        # Convert to DataFrames
        text_df = pd.DataFrame(text_features)
        num_df = pd.DataFrame(numerical_features).fillna(0)
        
        # Scale numerical features
        scaler = StandardScaler()
        self.scalers['numerical'] = scaler
        num_scaled = scaler.fit_transform(num_df)
        
        # Vectorize text features if requested
        if vectorize:
            print("Vectorizing text features...")
            # CSS code vectorization
            css_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
            css_features = css_vectorizer.fit_transform(text_df['css_code'])
            self.vectorizers['css'] = css_vectorizer
            
            # JS code vectorization
            js_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
            js_features = js_vectorizer.fit_transform(text_df['js_code'])
            self.vectorizers['js'] = js_vectorizer
            
            # Site type vectorization
            type_vectorizer = TfidfVectorizer(max_features=100)
            type_features = type_vectorizer.fit_transform(text_df['site_type'])
            self.vectorizers['type'] = type_vectorizer
            
            # Combine all features
            self.features = hstack([
                css_features, 
                js_features,
                type_features,
                csr_matrix(num_scaled)
            ])
            
            print(f"Feature matrix shape: {self.features.shape}")
        else:
            # Just use numerical features
            self.features = num_scaled
            
        print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        return self

    def reduce_dimensions(self, method='pca'):
        """Reduce dimensions of feature set"""
        print(f"Reducing dimensions using {method.upper()}...")
        start_time = time.time()
        
        if method == 'pca':
            reducer = PCA(n_components=self.n_components, random_state=self.random_state)
            if isinstance(self.features, csr_matrix):
                # Need to convert sparse matrix to dense for PCA
                self.features = self.features.toarray()
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        elif method == 'tsne':
            reducer = TSNE(n_components=min(self.n_components, 3), random_state=self.random_state)
            if isinstance(self.features, csr_matrix):
                self.features = self.features.toarray()
        else:
            raise ValueError(f"Unsupported dimension reduction method: {method}")
        
        # Fit and transform
        reduced_features = reducer.fit_transform(self.features)
        
        # Store reducer
        self.dimensionality_reducers[method] = reducer
        self.features = reduced_features
        
        # Normalize features
        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(self.features)
        self.scalers['final'] = scaler
        
        print(f"Reduced features shape: {self.features.shape}")
        print(f"Dimension reduction completed in {time.time() - start_time:.2f} seconds")
        
        # If using PCA, print explained variance
        if method == 'pca':
            explained_var = np.sum(reducer.explained_variance_ratio_)
            print(f"Explained variance with {self.n_components} components: {explained_var:.2%}")
        
        return self

    def visualize_clusters(self, labels, title="Cluster Visualization", save_path=None):
        """Visualize clusters in 2D using PCA or t-SNE"""
        # Ensure we have 2D data for plotting
        if self.features.shape[1] > 2:
            print("Reducing dimensions to 2D for visualization...")
            tsne = TSNE(n_components=2, random_state=self.random_state)
            features_2d = tsne.fit_transform(self.features)
        else:
            features_2d = self.features
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def evaluate_clusters(self, clustering_model, n_clusters):
        """Evaluate clustering quality with multiple metrics"""
        try:
            labels = clustering_model.labels_
            
            # Silhouette score (higher is better, range: -1 to 1)
            silhouette = silhouette_score(self.features, labels)
            
            # Calinski-Harabasz Index (higher is better)
            calinski = calinski_harabasz_score(self.features, labels)
            
            # Davies-Bouldin Index (lower is better)
            davies = davies_bouldin_score(self.features, labels)
            
            # Count number of samples in each cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_counts = dict(zip(unique_labels, counts))
            
            # Check if any cluster is too small or too large
            total_samples = len(labels)
            min_cluster_size = min(counts) / total_samples
            max_cluster_size = max(counts) / total_samples
            
            # If smallest cluster has less than 1% of data or largest has more than 90%, penalize
            balance_penalty = 0
            if min_cluster_size < 0.01 or max_cluster_size > 0.9:
                balance_penalty = 0.2
            
            # Custom score combining multiple metrics (normalize and weight)
            # Silhouette is in [-1,1], Davies-Bouldin is non-negative (lower is better)
            custom_score = silhouette - 0.1 * davies / (1 + davies) - balance_penalty
            
            results = {
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies,
                'custom_score': custom_score,
                'cluster_counts': cluster_counts
            }
            
            return results
        
        except Exception as e:
            print(f"Error evaluating clusters: {e}")
            return {
                'n_clusters': n_clusters,
                'silhouette': -1,
                'calinski_harabasz': 0,
                'davies_bouldin': 999,
                'custom_score': -999,
                'cluster_counts': {}
            }

    def tune_kmeans(self, min_clusters=2, max_clusters=20):
        """Find optimal number of clusters for K-Means"""
        print(f"Tuning K-Means (clusters range: {min_clusters}-{max_clusters})...")
        start_time = time.time()
        
        results = []
        best_score = -np.inf
        best_model = None
        best_n_clusters = 0
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            print(f"Testing K-Means with {n_clusters} clusters...")
            model = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=self.random_state
            )
            
            model.fit(self.features)
            evaluation = self.evaluate_clusters(model, n_clusters)
            results.append(evaluation)
            
            print(f"  Silhouette Score: {evaluation['silhouette']:.4f}")
            print(f"  Calinski-Harabasz: {evaluation['calinski_harabasz']:.4f}")
            print(f"  Davies-Bouldin: {evaluation['davies_bouldin']:.4f}")
            print(f"  Custom Score: {evaluation['custom_score']:.4f}")
            
            if evaluation['custom_score'] > best_score:
                best_score = evaluation['custom_score']
                best_model = model
                best_n_clusters = n_clusters
        
        # Store best model
        self.best_model = best_model
        self.best_n_clusters = best_n_clusters
        self.best_score = best_score
        
        print(f"Best K-Means model: {best_n_clusters} clusters with score {best_score:.4f}")
        print(f"K-Means tuning completed in {time.time() - start_time:.2f} seconds")
        
        # Visualize best clusters
        self.visualize_clusters(
            best_model.labels_,
            title=f"K-Means Clustering (k={best_n_clusters}, score={best_score:.4f})",
            save_path=self.output_dir / "kmeans_clusters.png"
        )
        
        # Save all results
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'cluster_counts'} 
            for r in results
        ])
        results_df.to_csv(self.output_dir / "kmeans_tuning_results.csv", index=False)
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['n_clusters'], results_df['silhouette'], 'o-', label='Silhouette Score')
        plt.axvline(x=best_n_clusters, color='r', linestyle='--', label=f'Best k={best_n_clusters}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('K-Means Clustering: Silhouette Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "kmeans_silhouette_analysis.png")
        
        return self

    def train_autoencoder(self, encoding_dim=None):
        """Train autoencoder for feature learning and dimensionality reduction"""
        print("Training autoencoder for feature learning...")
        start_time = time.time()
        
        # Set encoding dimension if not provided
        if encoding_dim is None:
            encoding_dim = min(self.n_components, 32)  # Default encoding dimension
        
        # Determine input dimensions
        input_dim = self.features.shape[1]
        
        # Define model architecture
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoder = Dense(input_dim // 2, activation='relu')(input_layer)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(input_dim // 4, activation='relu')(encoder)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(encoding_dim, activation='relu', name='encoder_output')(encoder)
        
        # Decoder
        decoder = Dense(input_dim // 4, activation='relu')(encoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(input_dim // 2, activation='relu')(decoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(input_dim, activation='linear')(decoder)
        
        # Create autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        
        # Create encoder model (for feature extraction)
        encoder_model = Model(inputs=input_layer, outputs=encoder)
        
        # Compile model
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                self.output_dir / "autoencoder_best.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with validation split
        history = autoencoder.fit(
            self.features, self.features,
            epochs=100,
            batch_size=32,
            shuffle=True,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save encoder (for feature extraction)
        encoder_model.save(self.output_dir / "encoder_model.h5")
        
        # Extract encoded features
        encoded_features = encoder_model.predict(self.features)
        
        # Update features with encoded version
        self.features = encoded_features
        self.best_encoder = encoder_model
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "autoencoder_training.png")
        
        print(f"Autoencoder training completed in {time.time() - start_time:.2f} seconds")
        print(f"Encoded features shape: {self.features.shape}")
        
        return self

    def build_combined_model(self):
        """Build and save the combined encoding+clustering model"""
        print("Building combined encoder + clustering model...")
        
        if self.best_encoder is None or self.best_model is None:
            raise ValueError("Train autoencoder and clustering model first")
        
        # Create the final pipeline model
        # This is saved as a set of separate components that can be loaded together
        components = {
            'encoder': self.best_encoder,
            'kmeans': self.best_model,
            'n_clusters': self.best_n_clusters,
            'kmeans_score': self.best_score,
            'scalers': self.scalers,
            'vectorizers': self.vectorizers,
            'dim_reducers': self.dimensionality_reducers,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': 'website_clustering'
        }
        
        # Save components
        joblib.dump(components, self.output_dir / "combined_model_components.joblib")
        
        # Save final trained model in .h5 format as requested
        self.best_encoder.save(self.output_dir / "final_model.h5")
        
        # Also save the K-means model separately
        joblib.dump(self.best_model, self.output_dir / "kmeans_model.joblib")
        
        # Save a simple text file with model parameters
        with open(self.output_dir / "model_info.txt", "w") as f:
            f.write(f"Website Clustering Model\n")
            f.write(f"Created: {components['created_at']}\n")
            f.write(f"Number of clusters: {self.best_n_clusters}\n")
            f.write(f"Silhouette score: {self.best_score:.4f}\n")
            f.write(f"Encoding dimensions: {self.features.shape[1]}\n")
            
        print(f"Combined model saved to {self.output_dir}")
        return self

    def analyze_clusters(self):
        """Analyze the characteristics of each cluster"""
        if self.best_model is None:
            raise ValueError("Train clustering model first")
        
        print(f"Analyzing {self.best_n_clusters} clusters...")
        
        # Get cluster labels
        labels = self.best_model.labels_
        
        # Create a DataFrame with features and labels
        feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]
        cluster_df = pd.DataFrame(self.features, columns=feature_names)
        cluster_df['cluster'] = labels
        
        # Add original site types
        site_types = [item.get('Type', '') for item in self.dataset]
        cluster_df['site_type'] = site_types
        
        # Calculate cluster centers
        cluster_centers = self.best_model.cluster_centers_
        
        # Analyze site types in each cluster
        cluster_analysis = {}
        for cluster_id in range(self.best_n_clusters):
            # Get sites in this cluster
            cluster_sites = cluster_df[cluster_df['cluster'] == cluster_id]
            
            # Count site types
            type_counts = {}
            for site_type in cluster_sites['site_type']:
                for t in site_type.split(','):
                    t = t.strip().lower()
                    if t:
                        type_counts[t] = type_counts.get(t, 0) + 1
            
            # Sort by count
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Store analysis
            cluster_analysis[cluster_id] = {
                'size': len(cluster_sites),
                'percentage': len(cluster_sites) / len(cluster_df) * 100,
                'top_types': sorted_types[:5],
                'center': cluster_centers[cluster_id]
            }
        
        # Save analysis
        with open(self.output_dir / "cluster_analysis.txt", "w") as f:
            f.write(f"Cluster Analysis for {self.best_n_clusters} clusters\n")
            f.write(f"Total samples: {len(cluster_df)}\n\n")
            
            for cluster_id, analysis in cluster_analysis.items():
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Size: {analysis['size']} sites ({analysis['percentage']:.2f}%)\n")
                f.write(f"  Top site types:\n")
                for site_type, count in analysis['top_types']:
                    f.write(f"    - {site_type}: {count} sites ({count/analysis['size']*100:.2f}%)\n")
                f.write("\n")
        
        print(f"Cluster analysis saved to {self.output_dir}/cluster_analysis.txt")
        return cluster_analysis

    def run_pipeline(self):
        """Run the full pipeline from data loading to model saving"""
        print("Running full clustering pipeline...")
        start_time = time.time()
        
        self.load_data()
        self.extract_features()
        self.reduce_dimensions('svd')  # SVD works better with sparse TF-IDF matrices
        self.train_autoencoder()
        self.tune_kmeans(min_clusters=5, max_clusters=20)
        self.build_combined_model()
        self.analyze_clusters()
        
        print(f"Pipeline completed in {time.time() - start_time:.2f} seconds")
        print(f"All outputs saved to {self.output_dir}")
        
        # Print final information
        print("\nFinal Model Summary:")
        print(f"  - Number of clusters: {self.best_n_clusters}")
        print(f"  - Silhouette score: {self.best_score:.4f}")
        print(f"  - Model saved at: {self.output_dir}/final_model.h5")
        
        return self

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train website clustering model')
    parser.add_argument('--data', required=True, help='Path to website JSON dataset')
    parser.add_argument('--output', default='model_output', help='Output directory for model and visualizations')
    parser.add_argument('--components', type=int, default=50, help='Number of components for dimension reduction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create and run the clustering model
    model = WebsiteClusteringModel(
        data_path=args.data,
        output_dir=args.output,
        n_components=args.components,
        random_state=args.seed
    )
    
    model.run_pipeline()

if __name__ == "__main__":
    main()
