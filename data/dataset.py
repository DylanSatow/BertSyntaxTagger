from torch.nn import Module, Linear
from transformers import BertModel

class SrlModel(Module):
    """BERT-based model for Semantic Role Labeling."""

    def __init__(self, num_labels):
        super(SrlModel, self).__init__()

        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = Linear(768, num_labels)

    def forward(self, input_ids, attn_mask, pred_indicator):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of token ids
            attn_mask: Attention mask for padding
            pred_indicator: Binary mask indicating predicate position
        
        Returns:
            logits: Classification logits for each token
        """
        bert_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=pred_indicator
        )

        enc_tokens = bert_output[0]
        logits = self.classifier(enc_tokens)

        return logits
