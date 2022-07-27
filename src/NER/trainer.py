import typing
import argparse
import pandas as pd
import torch

from tqdm import tqdm
from dataset import DataSet
from model import NerBert


# TODO: Split into train and test test.
class Trainer:
    def __init__(self, data: pd.DataFrame, settings: dict, model: NerBert=NerBert()):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.train_dataset = data #data.sample(frac=settings["train_ratio"])
        # self.test_dataset = data.drop(self.train_dataset.index)

        self.epochs = settings["epochs"]
        
        self.model = model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        self.model.train()
        
        train_dataset = DataSet(self.train_dataset)
        
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        for epoch in tqdm(range(self.epochs)):
            for batch in train_data_loader:
                self.optim.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)

                loss = outputs[0]
                loss.backward()
                self.optim.step()
