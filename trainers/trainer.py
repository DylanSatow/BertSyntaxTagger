import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


class SrlTrainer:
    def __init__(self, model, train_dataset, valid_dataset, config):
        """
        Initialize the trainer.
        
        Args:
            model: The SRL model
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            config: Training configuration
        """
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        self.config = config

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_predictions = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            ids = batch['ids'].squeeze(1).to(self.device)
            mask = batch['mask'].squeeze(1).to(self.device)
            targets = batch['targets'].squeeze(1).to(self.device)
            pred_mask = batch['pred'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(ids, mask, pred_mask)

            # Calculate loss
            loss = self.loss_fn(outputs.transpose(2, 1), targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=2)
            valid_mask = targets != -100
            correct = (predictions == targets) & valid_mask
            total_correct += correct.sum().item()
            total_predictions += valid_mask.sum().item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = total_correct / total_predictions if total_predictions > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}'
            })

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, path):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
