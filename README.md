# GutBrainIE RE Tasks
Pipeline for relation extraction over PubMed abstracts and titles using varying training data qualities and model architectures, including individual CNN-based models and ensemble.
## Directory Structure
```bash
├── configs/                     # YAML config files for Lightning CLI
│   ├── individual/             # Configs for individual models
│   └── ensemble/               # Configs for ensemble model
│
├── datamodule/
│   └── datamodule.py           # Handles data loading for training and evaluation
│
├── model/
│   ├── ensemble/               # Ensemble model definitions
│   │   ├── model.py
│   │   └── model_cnn.py
│   └── individual/             # Individual model definitions
│       ├── model.py
│       └── model_cnn.py
│
├── postprocess/
│   ├── evaluate.py             # Original evaluation script
│   ├── find_weights.py         # Finds optimal ensemble weights
│   ├── postprocessing.py       # Converts individual model output to final format
│   └── postprocessing_ensemble.py # Converts ensemble output to final format
│
├── preprocess/
│   ├── combineJSONFiles.py     # Combines multiple training datasets
│   ├── findSimilarSamples.py   # Finds similar samples using cosine similarity
│   └── preprocessing.py        # Main preprocessing for all datasets
│
├── run_ensemble.sh             #  script for running ensemble model
├── run_mixed_cnn.sh           #  script for mixed-quality CNN model
├── run_plat_plus_cnn.sh       #  script for Platinum+similargold CNN model
├── run_plat_plus_no_cnn.sh    #  script for Platinum+similargold non-CNN model
│
├── main.py                     # Entry point (Lightning CLI)
├── requirements.txt            # Python dependencies
```
## Requirements
- Python 3.11
- GPU (for training)
## Different Models
- run_plat_plus_cnn.sh – trains on Platinum + Some Gold (only those similar to what's in platinum) with CNN
- run_plat_plus_no_cnn.sh – trains on Platinum + Gold (only those similar to what's in platinum) without CNN
- run_mixed_cnn.sh – trains on Platinum + Gold + Silver with CNN
- run_ensemble.sh – ensemble consisting of a CNN model trained on Platinum + Gold + Silver, a CNN model trained on just Platinum + Gold, a regular model (deberta+linear layer) trained on Platinum + Gold + Silver
## Outputs
Final predictions are saved as:
- binary_tag_based_relations.json
- ternary_tag_based_relations.json
- ternary_mention_based_relations.json
## Results
TBD
## Notes
- The scripts I have provided will create a virtual environment in the project for you and automatically install dependencies
- We use Deberta to encode our sentences in all models :)
- Training multiple models at once is possible if they're on different GPUs, however not recommended because in the predict stage they write to the same file (depending on the model) so there is a small chance of conflict

