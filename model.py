import lightning as L
import torch
from transformers import DebertaV2Tokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import Trainer
import torch.nn.functional as F
import torch.nn as nn
import json

#Push to Binary RE Branch


#Deberta Model Class
class DeBertaModel(L.LightningModule): #added inheritance to lightning module here

        #Model Definition
        def __init__(self):

            super().__init__()

            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
            
            self.linear = nn.Linear(768, 6)
        def forward(self, input_ids, attention_mask):

            result = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
            result = self.linear(result)
            return result
            
        #Training Step
        def training_step(self, batch, batch_idx):

            input_ids = batch['input_ids']

            attention_mask = batch['attention_mask']

            labels = batch['labels']
            preds = self(input_ids, attention_mask)
            cls_toks = preds[:, 0, :]#using cls token - TODO: char is gonna fix this
            loss = F.cross_entropy(cls_toks, labels) 
            self.log('train_loss', loss)

            return loss


        #Optimizers
        def configure_optimizers(self):

            return torch.optim.AdamW(self.parameters(), lr=5e-5)


#Data Preperation Class
class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):

        self.texts = texts #sentences

        self.labels = labels #relation

        self.tokenizer = tokenizer

        self.max_len = max_len



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

        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'labels': torch.tensor(label, dtype=torch.long)

        }
def map_labels(labels): #map labels to ints (model works with numbers, not string labels :) )
    label_to_int = {}
    counter = 1
    for label in labels:
        if label not in label_to_int:
            label_to_int[label] = counter
            counter+=1
    return label_to_int

# Prepare data

# Open and read the JSON file

train_texts = []
train_labels = []

filename = sys.argv[1]

with open(filename, 'r') as file: 

    data = json.load(file)

for item in data:
    train_texts.append(item['sample'])
    train_labels.append(item['relation'])

label_to_int = map_labels(train_labels)

train_labels = [label_to_int[label] for label in train_labels] # all this is doing is turning the labels into their respective int

<<<<<<< HEAD
train_dataset = TextDataset(train_texts, train_labels, tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False),max_len=1000)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
=======
train_dataset = TextDataset(train_texts, train_labels, tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False), max_len=4096) #TODO: CHANGE MAX LENGTH

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) #TODO: change from 1 to 16 when it works
>>>>>>> f0f26de2dfe93b98165ca59d427d69f2c5e7aae1

# Initialize model

model = DeBertaModel()

# Train model

trainer = Trainer(max_epochs=3, precision="bf16-mixed") #TODO: keep precision, maybe increase GPUs if other two changes don't work out

trainer.fit(model, train_loader)
