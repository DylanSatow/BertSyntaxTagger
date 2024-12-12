import argparse
import torch
from data.dataset import SrlDataset
from data.utils import load_role_mappings, label_sentence
from models.srl_model import SrlModel
from configs.config import DataConfig
from evaluation.metrics import evaluate_spans, evaluate_spans_by_role
from transformers import BertTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SRL model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split', type=str, choices=['dev', 'test'], default='test',
                        help='Which dataset split to evaluate on')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save detailed results')
    return parser.parse_args()


def evaluate_model(model, data_loader, id_to_role, device):
    """Run full evaluation of the model."""
    # Overall metrics
    metrics = evaluate_spans(model, data_loader, id_to_role, device)

    # Per-role metrics
    role_metrics = evaluate_spans_by_role(
        model, data_loader, id_to_role, device)

    return metrics, role_metrics


def interactive_demo(model, tokenizer, id_to_role, device):
    """Run interactive demo for model predictions."""
    print("\nEntering interactive mode. Enter 'q' to quit.")

    while True:
        # Get sentence
        sentence = input("\nEnter a sentence (or 'q' to quit): ")
        if sentence.lower() == 'q':
            break

        tokens = sentence.split()

        # Get predicate index
        while True:
            try:
                print("\nTokens:", ' '.join(
                    f"{i}:{token}" for i, token in enumerate(tokens)))
                pred_idx = int(input("Enter the index of the predicate: "))
                if 0 <= pred_idx < len(tokens):
                    break
                print("Invalid index. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Get predictions
        labels = label_sentence(
            model, tokenizer, tokens, pred_idx,
            id_to_role, device
        )

        # Print results
        print("\nPredictions:")
        for token, label in zip(tokens, labels):
            print(f"{token}: {label}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configurations
    data_config = DataConfig()

    # Load role mappings
    role_to_id, id_to_role = load_role_mappings(data_config.role_list_file)
    num_labels = len(role_to_id) - 1  # Subtract 1 for [PAD]

    # Initialize model and load checkpoint
    model = SrlModel(num_labels=num_labels)
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)
    model.eval()

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        'bert-base-uncased', do_lower_case=True)

    # Load evaluation dataset
    eval_file = data_config.test_file if args.split == 'test' else data_config.dev_file
    eval_dataset = SrlDataset(eval_file, role_to_id)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Run evaluation
    print(f"\nEvaluating model on {args.split} set...")
    metrics, role_metrics = evaluate_model(
        model, eval_loader, id_to_role, device)

    # Print overall results
    print("\nOverall Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

    # Print per-role results
    print("\nResults by Role Type:")
    print(f"{'Role':<15} {'Precision':<10} {
          'Recall':<10} {'F1':<10} {'Support'}")
    print("-" * 55)

    for role, role_metric in sorted(role_metrics.items(), key=lambda x: x[1]['support'], reverse=True):
        print(f"{role:<15} {role_metric['precision']:.4f}     {role_metric['recall']:.4f}     "
              f"{role_metric['f1']:.4f}     {role_metric['support']}")

    # Save detailed results if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump({
                'overall_metrics': metrics,
                'role_metrics': role_metrics
            }, f, indent=2)
        print(f"\nDetailed results saved to {args.output_file}")

    # Run interactive demo
    interactive = input(
        "\nWould you like to try the interactive demo? (y/n): ")
    if interactive.lower() == 'y':
        interactive_demo(model, tokenizer, id_to_role, device)


if __name__ == '__main__':
    main()
