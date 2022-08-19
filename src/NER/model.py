import torch.nn as nn
import typing

from transformers import BertModel

class MZKBert(nn.Module):
    def __init__(self, num_labels: int):
        super(MZKBert, self).__init__()

        self.num_labels = num_labels
        
        self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.d0 = nn.Dropout(0.1)

        self.x1 = nn.Linear(768, 512)
        self.f1 = nn.ReLU()
        self.d1 = nn.Dropout(0.1)

        self.x2 = nn.Linear(512, 512)
        self.f2 = nn.ReLU()
        self.d2 = nn.Dropout(0.1)

        self.x3 = nn.Linear(512, self.num_labels)

    def forward(self, input_ids, attention_mask, labels):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        outputs = self.d0(bert_outputs[0])

        outputs = self.x1(outputs)
        outputs = self.f1(outputs)
        outputs = self.d1(outputs)

        outputs = self.x2(outputs)
        outputs = self.f2(outputs)
        outputs = self.d2(outputs)

        outputs = self.x3(outputs)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs.view(-1, self.num_labels), labels.view(-1))

        outputs = (outputs,) + bert_outputs[2:]

        return ((loss,) + outputs)
