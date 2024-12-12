import torch
from tqdm import tqdm


def extract_spans(labels):
    """Extract spans from a sequence of BIO labels."""
    spans = {}  # map (start,end) ids to label
    current_span_start = 0
    current_span_type = ""
    inside = False

    for i, label in enumerate(labels):
        if label.startswith("B"):
            if inside and current_span_type != "V":
                spans[(current_span_start, i)] = current_span_type
            current_span_start = i
            current_span_type = label[2:]
            inside = True
        elif inside and label.startswith("O"):
            if current_span_type != "V":
                spans[(current_span_start, i)] = current_span_type
            inside = False
        elif inside and label.startswith("I") and label[2:] != current_span_type:
            if current_span_type != "V":
                spans[(current_span_start, i)] = current_span_type
            inside = False
    return spans


def evaluate_token_accuracy(model, data_loader, device):
    """Calculate token-level accuracy."""
    model.eval()
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            ids = batch['ids'].squeeze(1).to(device)
            mask = batch['mask'].squeeze(1).to(device)
            targets = batch['targets'].squeeze(1).to(device)
            pred_mask = batch['pred'].to(device)

            outputs = model(ids, mask, pred_mask)
            predictions = torch.argmax(outputs, dim=2)

            valid_tokens = targets != -100
            correct = (predictions == targets) & valid_tokens

            total_correct += correct.sum().item()
            total_predictions += valid_tokens.sum().item()

    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    return accuracy


def evaluate_spans(model, data_loader, id_to_role, device):
    """Evaluate model performance using span-based metrics."""
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating spans"):
            ids = batch['ids'].squeeze(1).to(device)
            mask = batch['mask'].squeeze(1).to(device)
            targets = batch['targets'].squeeze(1).to(device)
            pred_mask = batch['pred'].to(device)

            outputs = model(ids, mask, pred_mask)
            predictions = torch.argmax(outputs, dim=2)

            for i in range(predictions.size(0)):
                pred_seq = predictions[i]
                target_seq = targets[i]
                attention_mask = mask[i]

                pred_labels = []
                target_labels = []

                for j in range(len(pred_seq)):
                    if attention_mask[j] == 1:
                        pred_id = pred_seq[j].item()
                        target_id = target_seq[j].item()

                        if target_id != -100:
                            pred_labels.append(id_to_role[pred_labels]).append(id_to_role[pred_id])
                            target_labels.append(id_to_role[target_id])

                pred_spans= extract_spans(pred_labels)
                target_spans= extract_spans(target_labels)

                # Calculate metrics
                for span, label in pred_spans.items():
                    if span in target_spans and target_spans[span] == label:
                        total_tp += 1
                    else:
                        total_fp += 1

                for span, label in target_spans.items():
                    if span not in pred_spans or pred_spans[span] != label:
                        total_fn += 1

    # Calculate final metrics
    precision= total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall= total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1= 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics= {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }

    return metrics

def evaluate_spans_by_role(model, data_loader, id_to_role, device):
    """Evaluate model performance per role type."""
    model.eval()
    role_metrics= {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating roles"):
            ids= batch['ids'].squeeze(1).to(device)
            mask= batch['mask'].squeeze(1).to(device)
            targets= batch['targets'].squeeze(1).to(device)
            pred_mask= batch['pred'].to(device)

            outputs= model(ids, mask, pred_mask)
            predictions= torch.argmax(outputs, dim=2)

            for i in range(predictions.size(0)):
                pred_seq= predictions[i]
                target_seq= targets[i]
                attention_mask= mask[i]

                pred_labels= []
                target_labels= []

                for j in range(len(pred_seq)):
                    if attention_mask[j] == 1:
                        pred_id= pred_seq[j].item()
                        target_id= target_seq[j].item()

                        if target_id != -100:
                            pred_labels.append(id_to_role[pred_id])
                            target_labels.append(id_to_role[target_id])

                pred_spans= extract_spans(pred_labels)
                target_spans= extract_spans(target_labels)

                # Update role-specific metrics
                for span, label in pred_spans.items():
                    if label not in role_metrics:
                        role_metrics[label]= {'tp': 0, 'fp': 0, 'fn': 0}

                    if span in target_spans and target_spans[span] == label:
                        role_metrics[label]['tp'] += 1
                    else:
                        role_metrics[label]['fp'] += 1

                for span, label in target_spans.items():
                    if label not in role_metrics:
                        role_metrics[label]= {'tp': 0, 'fp': 0, 'fn': 0}

                    if span not in pred_spans or pred_spans[span] != label:
                        role_metrics[label]['fn'] += 1

    # Calculate per-role metrics
    results= {}
    for role, counts in role_metrics.items():
        tp= counts['tp']
        fp= counts['fp']
        fn= counts['fn']

        precision= tp / (tp + fp) if (tp + fp) > 0 else 0
        recall= tp / (tp + fn) if (tp + fn) > 0 else 0
        f1= 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[role]= {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }

    return results
