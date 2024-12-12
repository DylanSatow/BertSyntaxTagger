def load_role_mappings(role_list_file):
    """
    Load role mappings from file.
    
    Args:
        role_list_file: Path to the file containing role list
        
    Returns:
        role_to_id: Dictionary mapping roles to IDs
        id_to_role: Dictionary mapping IDs to roles
    """
    role_to_id = {}
    with open(role_list_file, 'r') as f:
        role_list = [x.strip() for x in f.readlines()]
        role_to_id = dict((role, index)
                          for (index, role) in enumerate(role_list))
        role_to_id['[PAD]'] = -100

        id_to_role = dict((index, role)
                          for (role, index) in role_to_id.items())

    return role_to_id, id_to_role


def decode_predictions(outputs, id_to_role):
    """
    Convert model outputs to role labels.
    
    Args:
        outputs: Model output logits
        id_to_role: Mapping from IDs to role labels
        
    Returns:
        List of predicted role labels
    """
    import torch
    predictions = torch.argmax(outputs, dim=2)
    decoded = []

    for pred_id in predictions[0]:
        role = id_to_role.get(pred_id.item(), 'O')
        decoded.append(role)

    return decoded


def label_sentence(model, tokenizer, sentence, pred_idx, role_to_id, device):
    """
    Label a single sentence with semantic roles.
    
    Args:
        model: The SRL model
        tokenizer: BERT tokenizer
        sentence: List of tokens
        pred_idx: Index of the predicate
        role_to_id: Mapping from roles to IDs
        device: Torch device
        
    Returns:
        List of predicted labels for each token
    """
    model.eval()

    # Tokenize sentence
    bert_tokens = []
    token_map = []  # Maps original tokens to BERT token positions

    for i, word in enumerate(sentence):
        word_tokens = tokenizer.tokenize(word)
        bert_tokens.extend(word_tokens)
        token_map.extend([i] * len(word_tokens))

    # Truncate if needed
    if len(bert_tokens) > 126:  # Reserve space for [CLS] and [SEP]
        bert_tokens = bert_tokens[:126]
        token_map = token_map[:126]

    # Add special tokens
    bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']
    token_map = [-1] + token_map + [-1]

    # Convert to ids
    input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    padding = [tokenizer.pad_token_id] * (128 - len(input_ids))
    input_ids.extend(padding)

    # Create attention mask
    attention_mask = [1] * len(bert_tokens) + [0] * (128 - len(bert_tokens))

    # Create predicate mask
    pred_mask = [0] * 128
    bert_pred_positions = [i for i, x in enumerate(token_map) if x == pred_idx]
    if bert_pred_positions:
        pred_mask[bert_pred_positions[0]] = 1

    # Convert to tensors
    import torch
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor(
        [attention_mask], dtype=torch.long).to(device)
    pred_mask = torch.tensor([pred_mask], dtype=torch.long).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attn_mask=attention_mask, pred_indicator=pred_mask)

    # Decode predictions
    predictions = torch.argmax(outputs, dim=2)[0].cpu().numpy()
    final_labels = []

    for i, token in enumerate(sentence):
        # Find BERT tokens corresponding to this original token
        bert_indices = [j for j, x in enumerate(token_map) if x == i]
        if not bert_indices:
            final_labels.append("O")
            continue

        # Use prediction of first subword token
        pred_id = predictions[bert_indices[0]]
        if pred_id in id_to_role:
            label = id_to_role[pred_id]

            # Handle continuation of B- labels
            if (i > 0 and label.startswith('B-') and
                len(final_labels) > 0 and
                (final_labels[-1].startswith('B-') or final_labels[-1].startswith('I-')) and
                    final_labels[-1][2:] == label[2:]):
                label = 'I-' + label[2:]

            final_labels.append(label)
        else:
            final_labels.append("O")

    return final_labels
