#Dataset
import ast
import numpy as np
import os, sys
from PIL import Image
import pandas as pd
import pickle
import traceback
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')




def load_image_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

#from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel

class EmoticDatasetWithText(Dataset):
    def __init__(self, data_df, transform, pkl_file, scenario_file, data_src='./'):
        super(EmoticDatasetWithText, self).__init__()
        self.data_df = pd.read_csv(data_df) if isinstance(data_df, str) else data_df
        self.data_src = data_src
        self.transform = transform
        self.clip_features = self.load_image_data(pkl_file)
        self.scenario_data = self.load_scenario_data(scenario_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        #self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        #self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

        self.cat2ind = {emotion: idx for idx, emotion in enumerate([
            'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence',
            'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment',
            'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain',
            'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
        ])}
        self.ind2cat = {v: k for k, v in self.cat2ind.items()}

        # ind2vad
        self.ind2vad = {0: 'Valence', 1: 'Arousal', 2: 'Dominance'}

    def load_image_data(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def load_scenario_data(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        try:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            row = self.data_df.loc[index]
            bbox = ast.literal_eval(row['BBox'])
            bbox_face = ast.literal_eval(row['BBox_Face'])
            image_context, image_body, image_face, W, H = self.process_images(row, device, bbox, bbox_face)

            bbox_body = torch.tensor([bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H], dtype=torch.float32).to(device)
            bbox_face = torch.tensor([bbox_face[0]/(bbox[2]-bbox[0]), bbox_face[1]/(bbox[3]-bbox[1]), bbox_face[2]/(bbox[2]-bbox[0]), bbox_face[3]/(bbox[3]-bbox[1])], dtype=torch.float32).to(device)

            cat_labels = self.cat_to_one_hot(ast.literal_eval(row['Categorical_Labels']))
            cat_labels = torch.tensor(cat_labels, dtype=torch.float32).to(device)
            cont_labels = torch.tensor(ast.literal_eval(row['Continuous_Labels']), dtype=torch.float32).to(device) / 10.

            scenario = self.scenario_data.get(row['Filename'], {}).get('scenario', '')
            scenario_embedding = self.process_text(scenario).to(device) if scenario else None

            clip_context = torch.tensor(self.clip_features[row['Filename']]['features']).to(device)

            return {
                "context": image_context,
                "body": image_body,
                "face": image_face,
                "clip_context": clip_context,
                "bbox_body": bbox_body,
                "bbox_face": bbox_face,
                "cat_label": cat_labels,
                "cont_label": cont_labels,
                "filename": row['Filename'],
                "scenario": scenario_embedding
            }
        except Exception as e:
            logging.error(f"Error loading item {index}: {e}, Traceback: {traceback.format_exc()}")
            return None

    def cat_to_one_hot(self, cat):
        one_hot_cat = np.zeros(26)
        for em in cat:
            one_hot_cat[self.cat2ind[em]] = 1
        return one_hot_cat

    def process_images(self, row, device, bbox, bbox_face):
        image_path = os.path.join(self.data_src, row['Folder'], row['Filename'])
        image_context = Image.open(image_path).convert("RGB")
        W, H = image_context.size

        image_body = image_context.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        if bbox_face != [0, 0, 0, 0]:
            image_face = image_body.crop((bbox_face[0], bbox_face[1], bbox_face[2], bbox_face[3]))
        else:
            image_face = image_body

        image_context = self.transform(image_context.resize((224, 224)))
        image_body = self.transform(image_body.resize((224, 224)))
        image_face = self.transform(image_face.resize((112, 112)))

        return image_context.to(device), image_body.to(device), image_face.to(device), W, H

    def process_text(self, text):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.squeeze(0)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    new_batch = []
    for item in batch:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                new_item[k] = torch.tensor(v).to(device) 
            elif isinstance(v, torch.Tensor):
                new_item[k] = v.to(device) 
            else:
                new_item[k] = v 
        new_batch.append(new_item)
    return torch.utils.data.dataloader.default_collate(batch)



