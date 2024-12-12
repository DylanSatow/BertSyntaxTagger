import os
import argparse
import torch
from data.dataset import SrlDataset
from data.utils import load_role_mappings
from models.srl_model import SrlModel
from trainers.trainer import SrlTrainer
from configs.config import TrainingConfig, DataConfig
from evaluation.metrics import evaluate_spans


def parse_args():
    parser = argparse.ArgumentParser(description='Train SRL model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configs
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir
    )
    data_config = DataConfig()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load role mappings
    role_to_id, id_to_role = load_role_mappings(data_config.role_list_file)
    num_labels = len(role_to_id) - 1  # Subtract 1 for [PAD]

    # Initialize datasets
    train_dataset = SrlDataset(
        data_config.train_file,
        role_to_id,
        max_len=training_config.max_seq_length
    )
    dev_dataset = SrlDataset(
        data_config.dev_file,
        role_to_id,
        max_len=training_config.max_seq_length
    )

    # Initialize model
    model = SrlModel(num_labels=num_labels)

    # Initialize trainer
    trainer = SrlTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=dev_dataset,
        config=training_config
    )

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_f1 = 0.0

    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")

        # Train
        train_loss, train_acc = trainer.train_epoch()
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Evaluate
        metrics = evaluate_spans(
            model=model,
            data_loader=trainer.valid_loader,
            id_to_role=id_to_role,
            device=device
        )

        print(f"Validation Metrics:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")

        # Save checkpoint if better F1
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'srl_model_epoch_{epoch + 1}_f1_{best_f1:.4f}.pt'
            )
            trainer.save_checkpoint(epoch + 1, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
