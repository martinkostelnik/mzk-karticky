import torch

from transformers import BertForTokenClassification

# TODO: Change the num_labels parameter to be inferred, not hard-coded.
class NerBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bert = BertForTokenClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=65)

        
    def forward(self, input_id, attention_mask, labels):
        return self.bert(input_ids=input_id, attention_mask=attention_mask, labels=labels)
