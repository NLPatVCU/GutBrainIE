import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
from transformers import DebertaV2Tokenizer, DebertaV2TokenizerFast
class TrainDataset(Dataset):

    def __init__(self, path, max_len):
        texts = []
        labels = []
        spans = []

        data = None
        with open(path, 'r') as file: 
            data = json.load(file)

        for item in data:
            texts.append(item['sample'])
            labels.append(item['relation'])
            spans.append(((item["relative_subject_start"], item["relative_subject_end"]),(item["relative_object_start"], item["relative_object_end"])))



        label_to_int = {'NONE': 0, 'impact': 1, 'influence': 2, 'interact': 3, 'located in': 4, 'change expression': 5, 'target': 6, 'part of': 7, 'used by': 8, 'change abundance': 9, 'is linked to': 10, 'strike': 11, 'affect': 12, 'change effect': 13, 'produced by': 14, 'administered': 15, 'is a': 16, 'compared to': 17}

        labels = [label_to_int[label] for label in labels] # all this is doing is turning the labels into their respective int
        tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")


        self.texts = texts #sentences

        self.labels = labels #relation

        self.tokenizer = tokenizer

        self.max_len = max_len
        self.spans = spans


    def __len__(self):

        return len(self.texts)



    def __getitem__(self, idx):

        text = self.texts[idx]

        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(

            text,

            add_special_tokens=True,

            max_length=self.max_len,

            return_token_type_ids=False,

            padding='max_length',

            return_attention_mask=True,

            return_tensors='pt',

            truncation=True,

        )
        subject_start_token = encoding.char_to_token(self.spans[idx][0][0])
        subject_end_token = encoding.char_to_token(self.spans[idx][0][1])
        object_start_token = encoding.char_to_token(self.spans[idx][1][0])
        object_end_token = encoding.char_to_token(self.spans[idx][1][1])
        entity_mask = [0 for x in encoding['input_ids'].flatten()]
        for i in range(subject_start_token, subject_end_token+1):
            entity_mask[i] = 1
        for i in range(object_start_token, object_end_token+1):
            entity_mask[i]=2
        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'labels': torch.tensor(label, dtype=torch.long),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.long)

        }
class TestDataset(Dataset):

    def __init__(self, path, max_len):
        texts = []
        labels = []
        spans = []

        data = None
        with open(path, 'r') as file: 
            data = json.load(file)

        for item in data:
            texts.append(item['sample'])
            spans.append(((item["relative_subject_start"], item["relative_subject_end"]),(item["relative_object_start"], item["relative_object_end"])))


        tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")


        self.texts = texts #sentences

        self.tokenizer = tokenizer

        self.max_len = max_len
        self.spans = spans



    def __len__(self):

        return len(self.texts)



    def __getitem__(self, idx):

        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(

            text,

            add_special_tokens=True,

            max_length=self.max_len,

            return_token_type_ids=False,

            padding='max_length',

            return_attention_mask=True,

            return_tensors='pt',

            truncation=True,

        )
        subject_start_token = encoding.char_to_token(self.spans[idx][0][0])
        subject_end_token = encoding.char_to_token(self.spans[idx][0][1]+1)
        object_start_token = encoding.char_to_token(self.spans[idx][1][0])
        object_end_token = encoding.char_to_token(self.spans[idx][1][1]+1)
        entity_mask = [0 for x in encoding['input_ids'].flatten()]
        for i in range(subject_start_token, subject_end_token):
            entity_mask[i] = 1
        for i in range(object_start_token, object_end_token):
            entity_mask[i]=2

        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.long)


        }
class DataModule(L.LightningDataModule):
    def __init__(self, train_path="", val_path="", test_path="", batch_size=16, max_len=350):
        super().__init__()
        self.train_path = train_path
        self.val_path= val_path
        self.test_path = test_path
        self.batch_size=batch_size
        self.max_len=max_len
    def setup(self, stage=None):
        if stage=="fit" or stage is None:
            self.train_dataset = TrainDataset(self.train_path, self.max_len)
            self.val_dataset= TrainDataset(self.val_path, self.max_len)
        if stage=="predict" or stage is None:
            self.test_dataset=TestDataset(self.test_path, self.max_len)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
