from torch.utils.data import Dataset
import torch
from transformers import BertTokenizerFast


class SrlDataset(Dataset):
    """Dataset class for Semantic Role Labeling task."""

    def __init__(self, filename, role_to_id, max_len=128):
        super(SrlDataset, self).__init__()

        self.max_len = max_len
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.role_to_id = role_to_id
        self.items = self._load_data(filename)

    def _load_data(self, filename):
        """Load and preprocess the data from file."""
        items = []
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f]
            for sentence, labels in zip(lines[1::4], lines[3::4]):
                tokens_labels = self._tokenize_with_labels(
                    sentence.split(),
                    labels.split()
                )
                items.append(tokens_labels)
        return items

    def _tokenize_with_labels(self, sentence, text_labels):
        """Tokenize sentence while maintaining label alignment."""
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            word_tokens = self.tokenizer.tokenize(word)
            tokenized_sentence.append(word_tokens[0])
            labels.append(label)

            for tok in word_tokens[1:]:
                tokenized_sentence.append(tok)
                if label != 'O':
                    labels.append(f'I-{label[2:]}')
                else:
                    labels.append('O')

        return tokenized_sentence, labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tokens, labels = self.items[idx]
        s_len = len(tokens)

        if s_len > self.max_len - 2:  # -2 for [CLS] and [SEP]
            tokens = tokens[:(self.max_len - 2)]
            labels = labels[:(self.max_len - 2)]
            s_len = self.max_len - 2

        # Prepare input tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]'] + \
            (['[PAD]'] * (self.max_len - (s_len + 2)))
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Prepare labels
        labels = ['[CLS]'] + labels + ['[SEP]'] + \
            (['[PAD]'] * (self.max_len - (s_len + 2)))
        label_ids = []
        for label in labels:
            if label == '[PAD]':
                label_ids.append(self.role_to_id['[PAD]'])
            elif label not in self.role_to_id:
                label_ids.append(self.role_to_id['O'])
            else:
                label_ids.append(self.role_to_id[label])

        # Create attention mask
        attention_mask = ([1] * (s_len + 2)) + \
            ([0] * (self.max_len - (s_len + 2)))

        # Create predicate indicator mask
        predicate_mask = [0] * self.max_len
        if 'B-V' in labels:
            predicate_mask[labels.index('B-V')] = 1

        # Convert to tensors
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        predicate_mask = torch.tensor(predicate_mask, dtype=torch.long)

        return {
            'ids': token_ids.unsqueeze(0),
            'mask': attention_mask.unsqueeze(0),
            'targets': label_ids.unsqueeze(0),
            'pred': predicate_mask
        }
