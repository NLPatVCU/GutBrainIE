import pickle
import lightning as L
import torch
from transformers import DebertaV2Tokenizer, AutoModel, DebertaV2TokenizerFast
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import Trainer
import torch.nn.functional as F
import torch.nn as nn
import json
import sys
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torchmetrics
import matplotlib.pyplot as plt
# import wandb
wandb_logger = WandbLogger(project="GutBrainIE", name="one_epoch_test", log_model=True)
#Push to Binary RE Branch


#Deberta Model Class
class DeBertaModel(L.LightningModule): #added inheritance to lightning module here

        #Model Definition
        def __init__(self, class_weights, lr=5e-5,num_labels=18):

            super().__init__()

            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
            self.linear = nn.Linear(768*2, 18)
            self.lr = lr
            self.class_weights = class_weights
            self.save_hyperparameters()
            # Metrics for validation
            self.val_f1 = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average=None)
            self.val_f1_micro = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average='micro')
            self.val_precision = torchmetrics.Precision(num_classes=num_labels, task="multiclass", average=None)
            self.val_recall = torchmetrics.Recall(num_classes=num_labels, task="multiclass", average=None)
            self.val_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_labels, task="multiclass", normalize='true')

            # Metrics for testing
            self.test_f1 = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average=None)
            self.test_f1_micro = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average='micro')
            self.test_precision = torchmetrics.Precision(num_classes=num_labels, task="multiclass", average=None)
            self.test_recall = torchmetrics.Recall(num_classes=num_labels, task="multiclass", average=None)
            self.test_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_labels, task="multiclass", normalize='true')

        
        def forward(self, input_ids, attention_mask, entity_mask):
            result = self.model(input_ids, attention_mask=attention_mask).last_hidden_state  # Shape: (batch_size, seq_len, 768)

            entity1_mask = (entity_mask == 1).unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
            entity2_mask = (entity_mask == 2).unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

            entity_1 = result * entity1_mask  # Shape: (batch_size, seq_len, 768)
            entity_2 = result * entity2_mask  # Shape: (batch_size, seq_len, 768)

            entity_1 = entity_1.sum(dim=1) / entity1_mask.sum(dim=1)#.clamp(min=1)  # Shape: (batch_size, 768)
            entity_2 = entity_2.sum(dim=1) / entity2_mask.sum(dim=1)#.clamp(min=1)  # Shape: (batch_size, 768)

            result = torch.cat((entity_1, entity_2), dim=-1)  # Shape: (batch_size, 768 * 2)

            result = self.linear(result)  # Shape: (batch_size, 18)
    
            return result
          
        #Training Step
        def training_step(self, batch, batch_idx):

            input_ids = batch['input_ids']

            attention_mask = batch['attention_mask']
            
            entity_mask = batch["entity_mask"]
            labels = batch['labels']
            preds = self(input_ids, attention_mask, entity_mask)
            class_weights = self.class_weights.to(preds.device)
            loss = F.cross_entropy(preds, labels, weight=class_weights) 
            self.log('train_loss', loss)
            return loss
        def validation_step(self, batch, batch_idx):
            # this is the validation loop
            print(f"Validation step {batch_idx} started")
            input_ids = batch['input_ids']

            attention_mask = batch['attention_mask']
            
            entity_mask = batch["entity_mask"]
            labels = batch['labels']
            preds = self(input_ids, attention_mask, entity_mask)
            val_loss = F.cross_entropy(preds, labels) 

            # Log to check if the loss is calculated
            print(f"Validation loss for batch {batch_idx}: {val_loss.item()}")

            # Convert predictions and labels to the correct format for metric calculations
            preds_class = preds.argmax(dim=1)  # Get the predicted class indices

            # Update and log F1 score, precision, recall, and confusion matrix for validation
            self.val_f1.update(preds_class, labels)
            self.val_f1_micro.update(preds_class, labels)
            self.val_precision.update(preds_class, labels)
            self.val_recall.update(preds_class, labels)
            self.val_confusion_matrix.update(preds_class, labels)
            self.log("val_loss", val_loss)
            return val_loss

        def on_validation_epoch_end(self):
            # Compute all metrics
            f1_per_class = self.val_f1.compute()
            precision_vals = self.val_precision.compute()
            recall_vals = self.val_recall.compute()
            f1_micro = self.val_f1_micro.compute()

            # Log overall metrics
            self.log('val_avg_f1', f1_per_class.mean(), on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_f1_micro', f1_micro, on_epoch=True, prog_bar=True, sync_dist=True)

            # Log per-class metrics
            for i, (f1, prec, rec) in enumerate(zip(f1_per_class, precision_vals, recall_vals)):
                self.log(f'val_f1_class_{i}', f1, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f'val_precision_class_{i}', prec, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f'val_recall_class_{i}', rec, on_epoch=True, prog_bar=False, sync_dist=True)

            # Log confusion matrix
            fig, ax = self.val_confusion_matrix.plot(add_text=False)
            # wandb.log({'val_confusion_matrix': [wandb.Image(fig)]})
            plt.close(fig)

            # Reset metrics
            self.val_f1.reset()
            self.val_f1_micro.reset()
            self.val_precision.reset()
            self.val_recall.reset()
            self.val_confusion_matrix.reset()


        def predict_step(self, batch , batch_idx, dataloader_idx=0):
            preds =  self(batch["input_ids"], batch["attention_mask"], batch["entity_mask"])
            preds = torch.argmax(preds, dim=1)
            return preds

        #Optimizers
        def configure_optimizers(self):

            return torch.optim.AdamW(self.parameters(), lr=self.lr)


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
        subject_start_token = encoding.char_to_token(self.spans[idx][0][0])
        subject_end_token = encoding.char_to_token(self.spans[idx][0][1]+1)
        object_start_token = encoding.char_to_token(self.spans[idx][1][0])
        object_end_token = encoding.char_to_token(self.spans[idx][1][1]+1)
        entity_mask = [0 for x in encoding['input_ids'].flatten()]
        for i in range(subject_start_token, subject_end_token):
            entity_mask[i] = 1
        for i in range(object_start_token, object_end_token):
            entity_mask[i]=2
        test_ids = [encoding["input_ids"].flatten()[i] for i in range(len(entity_mask)) if entity_mask[i]==1 or entity_mask[i]==2]

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

def map_labels(labels): #map labels to ints (model works with numbers, not string labels :) )
    label_to_int = {}
    counter = 1
    label_to_int["NONE"] = 0
    for label in labels:
        if label not in label_to_int:
            label_to_int[label] = counter
            counter+=1
    dist = {label_to_int[x]:0 for x in label_to_int}
    for label in labels:
        dist[label_to_int[label]]+=1
    return label_to_int, dist
def get_max_len_sent(tokenizer, sents):
    tok_sents = tokenizer(sents, truncation=False)  # Tokenize sentences
    max_len = 0
    for sent in tok_sents["input_ids"]:  # Iterate over tokenized sequences
        if len(sent) > max_len:
            max_len = len(sent)
    return max_len
# Prepare data

# Open and read the JSON file
load_checkpoint = False
checkpoint=None
if len(sys.argv)==5:
    load_checkpoint = True
    checkpoint = sys.argv[4]
train_texts = []
train_labels = []
train_spans = []

val_texts = []
val_labels = []
val_spans = []

test_texts = []
test_spans = []

traindata = None
testdata = None
valdata=None
trainfile = sys.argv[1]
valfile = sys.argv[2]
testfile = sys.argv[3]
with open(trainfile, 'r') as file: 
    traindata = json.load(file)

for item in traindata:
    train_texts.append(item['sample'])
    train_labels.append(item['relation'])
    train_spans.append(((item["relative_subject_start"], item["relative_subject_end"]),(item["relative_object_start"], item["relative_object_end"])))

with open(valfile, 'r') as file: 
    valdata = json.load(file)

for item in valdata:
    val_texts.append(item['sample'])
    val_labels.append(item['relation'])
    val_spans.append(((item["relative_subject_start"], item["relative_subject_end"]),(item["relative_object_start"], item["relative_object_end"])))



with open(testfile, "r") as file:
    testdata = json.load(file)
for item in testdata:
    test_texts.append(item["sample"])
    test_spans.append(((item["relative_subject_start"], item["relative_subject_end"]),(item["relative_object_start"], item["relative_object_end"])))
#label_to_int, dist = map_labels(train_labels)

label_to_int = {'NONE': 0, 'impact': 1, 'influence': 2, 'interact': 3, 'located in': 4, 'change expression': 5, 'target': 6, 'part of': 7, 'used by': 8, 'change abundance': 9, 'is linked to': 10, 'strike': 11, 'affect': 12, 'change effect': 13, 'produced by': 14, 'administered': 15, 'is a': 16, 'compared to': 17}

train_labels = [label_to_int[label] for label in train_labels] # all this is doing is turning the labels into their respective int
class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels), dtype=torch.float)
val_labels = [label_to_int[label] for label in val_labels] # all this is doing is turning the labels into their respective int
#tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
max_len = max(get_max_len_sent(tokenizer, train_texts), get_max_len_sent(tokenizer, val_texts))+50 #some leeway
train_dataset = TrainDataset(train_texts, train_labels,train_spans, tokenizer = tokenizer,max_len=max_len)
val_dataset = TrainDataset(val_texts, val_labels, val_spans, tokenizer=tokenizer, max_len=max_len)
test_dataset = TestDataset(test_texts,test_spans, tokenizer=tokenizer, max_len=max_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
model = DeBertaModel(class_weights=class_weights)
trainer = Trainer(max_epochs=100, accelerator="gpu", precision="bf16-mixed", logger=wandb_logger, callbacks=[EarlyStopping(monitor="val_f1_micro", mode="max")]) #TODO: keep precision, maybe increase GPUs if other two changes don't work out
if load_checkpoint:
    model = DeBertaModel.load_from_checkpoint(checkpoint)
else:
    trainer.fit(model, train_loader, val_loader)
trainer = Trainer()
predictions = trainer.predict(model, test_loader)
print(predictions)
with open("predictions.pkl", "wb") as file:
    pickle.dump(predictions, file)
