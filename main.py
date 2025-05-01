# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from model import DeBertaModel
from datamodule import DataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import json
import sys
import yaml
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.class_weights", "model.class_weights", apply_on="instantiate")

def cli_main():
    cli = MyLightningCLI(
            DeBertaModel, 
            DataModule,
            save_config_kwargs={"overwrite": True}
    )  
    
if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
