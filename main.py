from lightning.pytorch.cli import LightningCLI

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.class_weights", "model.init_args.class_weights", apply_on="instantiate")

def cli_main():
    cli = MyLightningCLI(
            save_config_kwargs={"overwrite": True}
    )  
    
if __name__ == "__main__":
    cli_main()
