import lightning as L
import torch
from transformers import DebertaTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import Trainer
import json

#Push to Binary RE Branch


#Deberta Model Class
class DeBertaModel: 

        #Model Definition
        def __init__(self):

            super().__init__()

            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

            self.tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

        def forward(self, input_ids, attention_mask):

            return self.model(input_ids, attention_mask=attention_mask)

        #Training Step
        def training_step(self, batch, batch_idx):

            input_ids = batch['input_ids']

            attention_mask = batch['attention_mask']

            labels = batch['labels']

            outputs = self(input_ids, attention_mask)

            loss = outputs.loss

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


# Prepare data

# Open and read the JSON file

train_texts = []
train_labels = []

with open('trainData.json', 'r') as file:

    data = json.load(file)

for item in data:
    train_texts.append(item['sample'])
    train_labels.append(item['relation'])



train_dataset = TextDataset(train_texts, train_labels, tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base") , max_len=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model

model = DeBertaModel()

# Train model

trainer = Trainer(max_epochs=3)

trainer.fit(model, train_loader)
