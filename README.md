# BERT-based Semantic Role Labeling

This project implements a BERT-based Semantic Role Labeling (SRL) system that identifies and labels semantic arguments associated with predicates in text. The system treats SRL as a sequence labeling task using BIO tagging scheme. This project is really just practice for me to convert jupyter notebook code to full pytorch 
pipelines. 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DylanSatow/BertSyntaxTagger.git
cd srl_project
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

## Data

The project uses the Ontonotes 5.0 dataset for training and evaluation. The data should be organized as follows:
- `data/propbank_train.tsv`
- `data/propbank_dev.tsv`
- `data/propbank_test.tsv`
- `data/role_list.txt`

## Training

To train the model, run:
```bash
python scripts/train.py \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --checkpoint_dir checkpoints
```

Training arguments:
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--num_epochs`: Number of training epochs (default: 2)
- `--checkpoint_dir`: Directory to save model checkpoints (default: 'checkpoints')

## Evaluation

To evaluate a trained model, run:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pt \
    --split test \
    --output_file results.json
```

Evaluation arguments:
- `--checkpoint`: Path to model checkpoint (required)
- `--split`: Which dataset split to evaluate on ('dev' or 'test', default: 'test')
- `--batch_size`: Batch size for evaluation (default: 32)
- `--output_file`: Path to save detailed results (optional)

The evaluation script will:
1. Calculate overall precision, recall, and F1 scores
2. Provide per-role type metrics
3. Offer an interactive demo mode to try the model on custom sentences

## Project Structure

```
srl_project/
│
├── data/                     # Data handling
│   ├── dataset.py           # Dataset class
│   └── utils.py             # Data utilities
│
├── models/                   # Model implementations
│   └── srl_model.py         # SRL BERT model
│
├── trainers/                # Training logic
│   └── trainer.py           # Training loop
│
├── evaluation/              # Evaluation
│   └── metrics.py           # Evaluation metrics
│
├── configs/                 # Configurations
│   └── config.py           # Config classes
│
├── scripts/                 # Training/evaluation scripts
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
```

## Model Architecture

The model uses BERT (bert-base-uncased) for token encoding and adds a classification layer to predict BIO tags for each token. The architecture includes:
- BERT encoder for contextual representations
- Linear classification head
- Predicate position encoding using segment embeddings

## Performance

On the OntoNotes 5.0 test set, the model achieves:
- F1: ~0.82
- Precision: ~0.81
- Recall: ~0.83

## License

This project is intended for educational purposes only. The Ontonotes 5.0 dataset is provided for use in COMS 4705 only.