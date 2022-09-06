import torch
import typing

from transformers import BertModel


class MZKBert(torch.nn.Module):
    def __init__(self, num_labels: int, pretrained_bert_path="bert-base-multilingual-uncased"):
        super(MZKBert, self).__init__()

        self.num_labels = num_labels
        
        self.bert = BertModel.from_pretrained(pretrained_bert_path)
        self.d0 = torch.nn.Dropout(0.1)

        self.x1 = torch.nn.Linear(768, 512)
        self.f1 = torch.nn.ReLU()
        self.d1 = torch.nn.Dropout(0.1)

        self.x2 = torch.nn.Linear(512, 512)
        self.f2 = torch.nn.ReLU()
        self.d2 = torch.nn.Dropout(0.1)

        self.x3 = torch.nn.Linear(512, self.num_labels)

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
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs.view(-1, self.num_labels), labels.view(-1))

        outputs = (outputs,) + bert_outputs[2:]

        return loss, outputs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def get_device(self):
        return next(self.parameters()).device


def build_model(num_labels, model_path=None, pretrained_bert_path=None):
    model = MZKBert(num_labels=num_labels, pretrained_bert_path=pretrained_bert_path)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from '{model_path}'.")

    return model

