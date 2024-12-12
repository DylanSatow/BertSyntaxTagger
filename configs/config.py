from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 2
    max_seq_length: int = 128
    checkpoint_dir: str = "checkpoints"
    model_name: str = "bert-base-uncased"


@dataclass
class DataConfig:
    """Data configuration parameters."""
    train_file: str = "data/propbank_train.tsv"
    dev_file: str = "data/propbank_dev.tsv"
    test_file: str = "data/propbank_test.tsv"
    role_list_file: str = "data/role_list.txt"


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    hidden_size: int = 768  # BERT hidden size
    dropout: float = 0.1
