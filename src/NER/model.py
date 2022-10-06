import torch
import typing
import helper

from transformers import BertModel, AutoConfig


class MZKBert(torch.nn.Module):
    def __init__(self, bert, model_config=helper.ModelConfig()):
        super(MZKBert, self).__init__()

        self.num_labels = model_config.num_labels
        self.config = model_config

        self.bert = bert
        self.d0 = torch.nn.Dropout(0.1)

        self.x1 = torch.nn.Linear(768, 512)
        self.f1 = torch.nn.ReLU()
        self.d1 = torch.nn.Dropout(0.1)

        self.x2 = torch.nn.Linear(512, 512)
        self.f2 = torch.nn.ReLU()
        self.d2 = torch.nn.Dropout(0.1)

        self.x3 = torch.nn.Linear(512, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        outputs = self.d0(bert_outputs[0])

        outputs = self.x1(outputs)
        outputs = self.f1(outputs)
        outputs = self.d1(outputs)

        outputs = self.x2(outputs)
        outputs = self.f2(outputs)
        outputs = self.d2(outputs)

        outputs = self.x3(outputs)
        
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs.view(-1, self.num_labels), labels.view(-1))

        outputs = (outputs,) + bert_outputs[2:]

        return loss, outputs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def get_device(self):
        return next(self.parameters()).device


def build_model(tokenizer, model_path=None, pretrained_bert_path=helper.BERT_BASE_NAME, model_config=helper.ModelConfig()):
    if model_path is not None:
        config = AutoConfig.from_pretrained(helper.BERT_BASE_NAME)
        bert = BertModel(config)
        bert.resize_token_embeddings(len(tokenizer))

        model = MZKBert(bert, model_config)
        model.load_state_dict(torch.load(model_path))

        return model

    bert = BertModel.from_pretrained(pretrained_bert_path)
    bert.resize_token_embeddings(len(tokenizer))
    model = MZKBert(bert, model_config)
    return model
