from safe_gpu.safe_gpu import GPUOwner
gpu_owner = GPUOwner()

import os
import sys
import argparse
import torch

from transformers import BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", help="Path to a file with training lines.")
    parser.add_argument("--test-path", help="Path to a file with testing lines.")
    parser.add_argument("--bert-path", help="Path to the model checkpoint.")
    parser.add_argument("--tokenizer-path", help="Path to the tokenizer checkpoint.")
    parser.add_argument("--save-dir", help="Path to directory where checkpoints are stored.")

    parser.add_argument("--batch-size", help="Batch size.", default=8, type=int)
    parser.add_argument("--learning-rate", help="Learning rate.", default=2e-5, type=float)
    parser.add_argument("--masking-prob", help="Masking probability.", default=0.15, type=float)

    parser.add_argument("--iterations", help="Number of training steps.", default=100000, type=int)
    parser.add_argument("--view-step", help="The number of iterations after which the model is tested.", default=100000, type=int)

    args = parser.parse_args()
    return args


class TextDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self._data.items()}
        return item

    def __len__(self):
        return len(self._data['input_ids'])


class Model:
    def __init__(self, checkpoint_path):
        self._model = BertForMaskedLM.from_pretrained(checkpoint_path)

    def to(self, device):
        self._model = self._model.to(device)
        return self

    def eval(self):
        self._model = self._model.eval()
        return self

    def train(self):
        self._model = self._model.train()
        return self

    def forward(self, batch):
        device = self.get_device()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = self._model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs.loss, outputs.logits

    def get_device(self):
        return next(self.parameters()).device

    def parameters(self):
        return self._model.parameters()

    def save(self, save_dir, iteration):
        path = os.path.join(save_dir, f"model-{iteration:06d}")
        self._model.save_pretrained(path)


class Trainer:
    def __init__(self, model, learning_rate):
        self._model = model
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=learning_rate)

    def train_step(self, batch):
        self._optimizer.zero_grad()
        loss, logits = self._model.forward(batch)
        loss.backward()
        self._optimizer.step()
        
        return loss.item(), logits.detach()

    def test_step(self, batch):
        with torch.no_grad():
            loss, logits = self._model.forward(batch)

        return loss.item(), logits.detach()

    def train(self, iterations, train_dataset, test_dataset, view_step=10000, save_dir=None):
        step = 0

        train_loss_accumulator = 0
        test_loss_accumulator = 0

        while step < iterations:
            for train_batch in train_dataset:
                train_loss, _ = self.train_step(train_batch)
                train_loss_accumulator += train_loss
                step += 1

                if step % view_step == 0:
                    self._model = self._model.eval()
                    for test_batch in test_dataset:
                        test_loss, _ = self.test_step(test_batch)
                        test_loss_accumulator += test_loss

                    print(f"Iteration:{step} train_loss:{train_loss_accumulator / view_step} test_loss:{test_loss_accumulator / view_step}")

                    train_loss_accumulator = 0
                    test_loss_accumulator = 0

                    self._model = self._model.train()

                    if save_dir is not None:
                        self._model.save(save_dir, step)

                if step == iterations:
                    break


def load_texts(path, min_length=10):
    lines = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) > min_length:
                lines.append(line)

    return lines


def load_data(train_path, test_path, batch_size, tokenizer_checkpoint, masking_prob=0.15):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=masking_prob, return_tensors="pt")
    print("Tokenizer and DataCollator created.")

    train_texts = load_texts(path=train_path)
    test_texts = load_texts(path=test_path)
    print(f"Texts loaded ({len(train_texts)}/{len(test_texts)} lines).")

    all_texts = train_texts + test_texts
    all_data = tokenizer(all_texts, truncation=True, padding=True)
    print("Texts tokenized.")

    train_data = {key: torch.tensor(val[:len(train_texts)]) for key, val in all_data.items()}
    test_data = {key: torch.tensor(val[len(train_texts):]) for key, val in all_data.items()}
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    print("Datasets created.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=data_collator)
    print("DataLoaders created.")

    return train_loader, test_loader


def load_model(model_checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(model_checkpoint)
    model = model.to(device)
    print("Model loaded.")

    return model


def main():
    args = parse_arguments()
    
    train_dataset, test_dataset = load_data(train_path=args.train_path, test_path=args.test_path, batch_size=args.batch_size, tokenizer_checkpoint=args.tokenizer_path, masking_prob=args.masking_prob)
    
    model = load_model(model_checkpoint=args.bert_path)

    trainer = Trainer(model, learning_rate=args.learning_rate)
    trainer.train(args.iterations, train_dataset, test_dataset, view_step=args.view_step, save_dir=args.save_dir)

    return 0


if __name__ == "__main__":
    exit(main())
