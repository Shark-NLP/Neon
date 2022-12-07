# -*- coding: utf-8 -*-


import os
import sys
import csv
import numpy as np
# import ujson as json
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm


class DataReader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = defaultdict(list)
        self.cnt = 0

    def tokenize(self, text):
        token_id = self.tokenizer.encode(text, add_special_tokens=False)
        bert_tokenized = self.tokenizer.decode(token_id)
        return token_id, bert_tokenized

    def load(self, file_or_example: str):
        # if dataloader is not None:
        #     assert isinstance(dataloader, MultiDataLoader) and role is not None
        #     data = dataloader.load_data(role)
        # else:
        if os.path.exists(file_or_example):
            with open(file_or_example, 'r', encoding='utf-8') as file:
                lines = list(csv.reader(file))
            lines = lines[1:501]

        for line in tqdm(lines):
            self.cnt += 1
            self._load(line)

        for k in self.data:
            assert self.cnt == len(self.data[k]), k

    def _load(self, example):
        '''
        NOTE: ADD a blank before all texts except for premise!!!
        :param example: {k: v, k: v}
        :return: {k: [{}], 'original_ending': [[{}, {}, {}], []]}
        '''
        for k in ['premise', 'contradiction']:
            if k == 'premise':
                text = example[0].strip()
            else:
                text = example[2].strip()
            token_id, bert_tokenized = self.tokenize(text)
            self.data[k].append({
                'token_ids': np.array(token_id),
                'text': bert_tokenized
            })

    def append_data(self, x1, x2):
        x1['text'] = x1['text'] + x2['text']
        x1['token_ids'] = self.tokenizer.encode(x1['text'], add_special_tokens=False)
        return x1

    def __len__(self):
        return self.cnt
