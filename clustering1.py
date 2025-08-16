import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import ParameterGrid
from transformers import AutoTokenizer, AutoModel
import torch
import h5py
import os
from tqdm import tqdm
import time


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states * attention_mask.unsqueeze(-1)
    sum_embeddings = torch.sum(last_hidden, dim=1)
    sum_mask = torch.clamp(torch.sum(attention_mask, dim=1).unsqueeze(-1), min=1e-9)
    return torch.nn.functional.normalize(sum_embeddings / sum_mask, p=2, dim=1)


def generate_embeddings(texts, model_name="microsoft/codebert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(texts), 8), desc="Generating embeddings"):
        batch = [text if text.strip() else " " for text in texts[i:i + 8]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.extend(average_pool(outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy())
    return np.array(embeddings)


def evaluate_clustering(X, labels):
    mask = (labels != -1) if -1 in labels else slice(None)
    return silhouette_score(X[mask], labels[mask]) if len(np.unique(labels[mask])) > 1 else -1


def grid_search(X, configs):
    best = {'score': -1, 'model': None, 'params': None}
    for name, cfg in configs.items():
        for params in ParameterGrid(cfg['params']):
            try:
                model = cfg['model'](**params)
                labels = model.fit_predict(X)
                score = evaluate_clustering(X, labels)
                if score > best['score']:
                    best.update({'model': model, 'score': score, 'params': params, 'name': name})
            except:
                continue
    return best


def save_model(result, css_emb, js_emb, scaler, mlb, file_path='website_cluster_model.h5'):
    with h5py.File(file_path, 'w') as hf:
        model_grp = hf.create_group('model')
        model_grp.attrs.update({
            'algorithm': result['name'],
            'params': str(result['params']),
            'silhouette_score': result['score']
        })
        hf.create_dataset('css_embeddings', data=css_emb)
        hf.create_dataset('js_embeddings', data=js_emb)

        scaler_grp = hf.create_group('scaler')
        scaler_grp.create_dataset('mean', data=scaler.mean_)
        scaler_grp.create_dataset('scale', data=scaler.scale_)

        if mlb.classes_ is not None:
            hf.create_dataset('mlb_classes', data=mlb.classes_.astype('S'))

        hf.create_dataset('labels', data=result['model'].labels_ if hasattr(result['model'], 'labels_') else result[
            'model'].predict(X))


def main(dataset_path):
    print("🚀 Starting website clustering pipeline...")

    data = load_dataset(dataset_path)
    css = [d['css-code'] for d in data]
    js = [d['js-code'] for d in data]
    types = [d['Type'].split(', ') for d in data]

    print("\n🔨 Generating CSS embeddings...")
    css_emb = generate_embeddings(css)
    print("✅ CSS embeddings completed!")

    print("\n🔨 Generating JS embeddings...")
    js_emb = generate_embeddings(js)
    print("✅ JS embeddings completed!")

    X = np.hstack([css_emb, js_emb])
    X = StandardScaler().fit_transform(X)

    print("\n🔍 Optimizing clustering parameters...")
    # Corrected parameter grids with proper list formatting
    best = grid_search(X, {
        'KMeans': {
            'model': KMeans,
            'params': {
                'n_clusters': [3, 5, 7, 9],
                'n_init': [10],  # Fixed as list
                'random_state': [42]  # Fixed as list
            }
        },
        'DBSCAN': {
            'model': DBSCAN,
            'params': {
                'eps': [0.3, 0.5, 0.7],
                'min_samples': [3, 5]
            }
        },
        'Agglomerative': {
            'model': AgglomerativeClustering,
            'params': {
                'n_clusters': [3, 5, 7],
                'linkage': ['ward', 'average']
            }
        }
    })

    print(f"\n🎉 Best model: {best['name']} (Score: {best['score']:.2f})")
    mlb = MultiLabelBinarizer().fit(types)
    save_model(best, css_emb, js_emb, StandardScaler().fit(X), mlb)
    print("💾 Model saved as 'website_cluster_model.h5'")


if __name__ == "__main__":
    main("website_dataset.json")
    print("\n✨ Pipeline completed successfully! ✨")
