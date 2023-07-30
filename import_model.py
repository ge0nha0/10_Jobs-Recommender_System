import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import gluonnlp as nlp

class pretrained_model:

    def __init__(self, args):
        self.vocab = None
        self.model = None
        self.tokenizer = None
        self.model_version = args.model_version

    def import_model(self):
        if self.model_version == 'original':
            self.device = torch.device("cuda:0")
            self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
            self.model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
            self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(self.tokenizer.vocab_file, padding_token='[PAD]')
        elif self.model_version == 'fine_tuned':
            self.device = torch.device("cuda:0")
            self.tokenizer = KoBERTTokenizer.from_pretrained('model_result')
            self.model = BertModel.from_pretrained('model_result', return_dict=False)
            self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(self.tokenizer.vocab_file, padding_token='[PAD]')