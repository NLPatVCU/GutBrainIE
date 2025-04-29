# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from model import DeBertaModel
from datamodule import DataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


def cli_main():
    cli = LightningCLI(
            DeBertaModel, 
            DataModule,
            save_config_kwargs={"overwrite": True}
    )  

if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
