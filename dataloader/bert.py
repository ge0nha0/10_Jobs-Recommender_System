from .base import Dataloader
import torch
import torch.utils.data as data_utils

import gluonnlp as nlp
import numpy as np
from torch.utils.data import Dataset
from import_model import pretrained_model


class BertDataloader(Dataloader):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.model = pretrained_model(args)
        self.model.import_model()

    def get_dataloaders(self):
        train_loader =  self._get_dataloader(self.train,self.args.train_batch_size)
        val_loader = self._get_dataloader(self.val,self.args.val_batch_size)
        test_loader = self._get_dataloader(self.test,self.args.test_batch_size)
        return  train_loader, val_loader, test_loader

    def _get_dataloader(self, dataset, batch_size):
        dataset = self._get_dataset(dataset)
        dataloader = data_utils.DataLoader(batch_size=batch_size, dataset=dataset, shuffle=False)
        return dataloader

    def _get_dataset(self, dataset):
        dataset = BERTDataset(self.args, dataset, self.model.tokenizer, self.model.vocab)
        return dataset
    

class BERTDataset(Dataset):

    def __init__(self, args, dataset, bert_tokenizer, vocab ):
      self.sent_idx = args.sent_idx
      self.label_idx = args.label_idx
      self.max_len = args.max_len
      self.pad = args.pad
      self.pair = args.pair

      transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=self.max_len, vocab=vocab, pad=self.pad, pair=self.pair)

      self.sentences = [transform([i[self.sent_idx]]) for i in dataset]
      self.labels = [np.int(i[self.label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))