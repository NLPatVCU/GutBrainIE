import lightning as L
import torch
from transformers import DebertaV2Tokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import Trainer
import torch.nn.functional as F
import torch.nn as nn
import json
import sys

#Push to Binary RE Branch


#Deberta Model Class
class DeBertaModel(L.LightningModule): #added inheritance to lightning module here

        #Model Definition
        def __init__(self):

            super().__init__()

            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
            self.linear = nn.Linear(768, 18)
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
            print(loss)
            return loss
        def predict_step(self, batch , batch_idx, dataloader_idx=0):
            preds =  self(batch["input_ids"], batch["attention_mask"])
            preds = preds[:, 0, :]#using cls token - TODO: char is gonna fix this
            preds = torch.argmax(preds, dim=1)
            return preds

        #Optimizers
        def configure_optimizers(self):

            return torch.optim.AdamW(self.parameters(), lr=5e-5)


#Data Preperation Class
class TrainDataset(Dataset):

    def __init__(self, texts, labels,spans, tokenizer, max_len):

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
        subject_start_token = encoding.char_to_token(self.spans[0][0])
        subject_end_token = encoding.char_to_token(self.spans[0][1]+1)
        object_start_token = encoding.char_to_token(self.spans[1][0])
        object_end_token = encoding.char_to_token(self.spans[1][1]+1)
        entity_mask = [0 for x in encoding['input_ids'].flatten()]
        for i in range(subject_start_token, subject_end_token+1):
            entity_mask[i] = 1
        for i in range(object_start_token, object_end_token+1):
            entity_mask[i]=2
        test_ids = [encoding["input_ids".flatten()][i] for i in range(len(entity_mask)) if entity_mask[i]==1 or entity_mask[i]==2]
        print(f"decoded: {self.tokenizer.decode(test_ids)}")
        print(text)
        print(text[subject_start_token:subject_end_token])
        print(text[object_start_token:object_end_token])
        exit()

        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'labels': torch.tensor(label, dtype=torch.long),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.long)

        }
class TestDataset(Dataset):

    def __init__(self, texts, spans, tokenizer, max_len):

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
        subject_start_token = encoding.char_to_token(self.spans[0][0])
        subject_end_token = encoding.char_to_token(self.spans[0][1]+1)
        object_start_token = encoding.char_to_token(self.spans[1][0])
        object_end_token = encoding.char_to_token(self.spans[1][1]+1)
        entity_mask = [0 for x in encoding['input_ids'].flatten()]
        for i in range(subject_start_token, subject_end_token+1):
            entity_mask[i] = 1
        for i in range(object_start_token, object_end_token+1):
            entity_mask[i]=2

        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.long)


        }

def map_labels(labels): #map labels to ints (model works with numbers, not string labels :) )
    label_to_int = {}
    counter = 1
    label_to_int["NONE"] = 0
    for label in labels:
        if label not in label_to_int:
            label_to_int[label] = counter
            counter+=1
    return label_to_int
def get_max_len_sent(tokenizer, sents):
    tok_sents = tokenizer(sents, truncation=False)
    max_len = 0
    for sent in tok_sents:
        if len(sent)>max_len:
            max_len = len(sent)
    return max_len


# Prepare data

# Open and read the JSON file
load_checkpoint = False
checkpoint=None
if len(sys.argv)==4:
    load_checkpoint = True
    checkpoint = sys.argv[3]
    train_texts = []
train_labels = []
train_spans = []
test_texts = []
test_spans = []

traindata = None
testdata = None

trainfile = sys.argv[1]
testfile = sys.argv[2]
with open(trainfile, 'r') as file: 

    traindata = json.load(file)

for item in traindata:
    train_texts.append(item['sample'])
    train_labels.append(item['relation'])
    train_spans.append(((item["relative_subject_start"], item["relative_subject_end"]),(item["relative_object_start"], item["relative_object_end"])))
with open(testfile, "r") as file:
    testdata = json.load(file)
for item in testdata:
    test_texts.append(item["sample"])
    test_spans.append(((item["relative_subject_start"], item["relative_subject_end"]),(item["relative_object_start"], item["relative_object_end"])))
label_to_int = map_labels(train_labels)

print(label_to_int)

train_labels = [label_to_int[label] for label in train_labels] # all this is doing is turning the labels into their respective int
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
max_len = get_max_len_sent(tokenizer, train_texts)+50 #just some leeway
train_dataset = TrainDataset(train_texts, train_labels,train_spans, tokenizer = tokenizer,max_len=max_len)
test_dataset = TestDataset(test_texts,test_spans, tokenizer=tokenizer, max_len=max_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
model = DeBertaModel()
trainer = Trainer(max_epochs=1, accelerator="gpu", precision="bf16-mixed") #TODO: keep precision, maybe increase GPUs if other two changes don't work out
if load_checkpoint:
    model = DeBertaModel.load_from_checkpoint(checkpoint)
    trainer = Trainer()
else:
    trainer.fit(model, train_loader)
predictions = trainer.predict(model, test_loader)
print(predictions)
