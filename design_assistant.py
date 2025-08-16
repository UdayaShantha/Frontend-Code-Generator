import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import json


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states * attention_mask.unsqueeze(-1)
    sum_embeddings = torch.sum(last_hidden, dim=1)
    sum_mask = torch.clamp(torch.sum(attention_mask, dim=1).unsqueeze(-1), min=1e-9)
    return torch.nn.functional.normalize(sum_embeddings / sum_mask, p=2, dim=1)


class DesignAssistant:
    def __init__(self, model_path='website_cluster_model.h5', dataset_json='website_dataset.json'):
        print(f"Loading model and data from '{model_path}' ...")
        with h5py.File(model_path, 'r') as hf:
            self.scaler_mean = hf['scaler/mean'][:]
            self.scaler_scale = hf['scaler/scale'][:]

            self.css_embeddings = hf['css_embeddings'][:]
            self.js_embeddings = hf['js_embeddings'][:]

            if 'labels' not in hf:
                raise ValueError("labels dataset missing in the HDF5 file.")
            self.labels = hf['labels'][:]

            if 'model/cluster_centers' in hf:
                self.cluster_centers = hf['model/cluster_centers'][:]
            else:
                self.cluster_centers = None

        # Load CSS and JS code from JSON dataset file (external to .h5)
        print(f"Loading CSS and JS code from dataset JSON '{dataset_json}' ...")
        with open(dataset_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            self.css_code = [item.get('css-code', '') for item in data]
            self.js_code = [item.get('js-code', '') for item in data]

        # Check lengths match
        if len(self.css_code) != len(self.css_embeddings) or len(self.js_code) != len(self.js_embeddings):
            raise ValueError("Mismatch between code snippets and embeddings count.")

        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.code_model = AutoModel.from_pretrained('microsoft/codebert-base').to(self.device)
        self.code_model.eval()

        self.image_model = models.resnet50(pretrained=True).to(self.device)
        self.image_model.eval()
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        combined_emb = np.hstack([self.css_embeddings, self.js_embeddings])
        self.nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn.fit(combined_emb)
        print("Model and data loaded successfully.")

    def text_to_embedding(self, text_prompt):
        inputs = self.tokenizer(text_prompt, return_tensors="pt",
                                padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.code_model(**inputs)
        emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        return emb.cpu().numpy()

    def image_to_embedding(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_t = self.image_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.image_model(img_t).cpu().numpy()

        # Placeholder projection: zero pad to combined embedding size
        combined_dim = self.css_embeddings.shape[1] + self.js_embeddings.shape[1]
        projected = np.zeros((1, combined_dim))
        projected[0, :features.shape[1]] = features
        return projected

    def process_input(self, text_prompt=None, image_path=None):
        if text_prompt:
            emb = self.text_to_embedding(text_prompt)
            emb = np.hstack([emb, emb])  # duplicate to match combined dim
        elif image_path:
            emb = self.image_to_embedding(image_path)
        else:
            raise ValueError("Either text_prompt or image_path must be provided.")

        emb_scaled = (emb - self.scaler_mean) / self.scaler_scale
        distances, indices = self.nn.kneighbors(emb_scaled)
        best_idx = indices[0][0]

        return {
            'css_code': self.css_code[best_idx],
            'js_code': self.js_code[best_idx],
            'cluster_label': int(self.labels[best_idx])
        }


if __name__ == "__main__":
    assistant = DesignAssistant(model_path='website_cluster_model.h5', dataset_json='website_dataset.json')

    # Example text prompt
    prompt = "Create a clean, educational website with blue accents and smooth scrolling"
    print("\nProcessing text prompt...")
    result = assistant.process_input(text_prompt=prompt)
    print("\n=== Generated CSS ===\n", result['css_code'])
    print("\n=== Generated JavaScript ===\n", result['js_code'])
    print(f"\nCluster label: {result['cluster_label']}")

    # Example image input (uncomment and provide a valid image path)
    # image_file = "your_image.png"
    # print("\nProcessing image input...")
    # result_img = assistant.process_input(image_path=image_file)
    # print("\n=== Generated CSS ===\n", result_img['css_code'])
    # print("\n=== Generated JavaScript ===\n", result_img['js_code'])
    # print(f"\nCluster label: {result_img['cluster_label']}")
