import pickle
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import Trainer
import torch.nn.functional as F
import torch.nn as nn
import json
import sys
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torchmetrics
import matplotlib.pyplot as plt
from transformers import AutoModel
import wandb
#Deberta Model Class
class DeBertaModel(L.LightningModule): #added inheritance to lightning module here

        #Model Definition
        def __init__(self, class_weights, lr=5e-5,num_labels=18):

            super().__init__()

            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
            self.linear = nn.Linear(768*2, 18)
            self.lr = lr
            self.class_weights = torch.Tensor(class_weights)
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
            self.preds = []
        
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
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', current_lr, prog_bar=True, logger=True, on_step=True)
            return loss
        def validation_step(self, batch, batch_idx):
            # this is the validation loop
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
            # fig, ax = self.val_confusion_matrix.plot(add_text=True)
            # wandb.log({'val_confusion_matrix': [wandb.Image(fig)]})
            # plt.close(fig)
            
            cm = self.val_confusion_matrix.compute().cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, cmap='Blues')
            plt.colorbar(im)
            plt.title("Normalized Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            wandb.log({'val_confusion_matrix': wandb.Image(fig)})
            plt.close(fig)


            # Reset metrics
            self.val_f1.reset()
            self.val_f1_micro.reset()
            self.val_precision.reset()
            self.val_confusion_matrix.reset()
            self.val_recall.reset()
            self.val_confusion_matrix.reset()


        def predict_step(self, batch , batch_idx, dataloader_idx=0):
            preds =  self(batch["input_ids"], batch["attention_mask"], batch["entity_mask"])
            #preds = torch.argmax(preds, dim=1)
            self.preds.append(preds.cpu().numpy())
            with open("predictions.pkl", "wb") as f:
                pickle.dump(self.preds, f)
            return preds

        #Optimizers
        def configure_optimizers(self):

            optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.trainer.estimated_stepping_batches)
            return {"optimizer": optim, "lr_scheduler":{"scheduler": scheduler,"interval":"step", "frequency":1}}



    
