from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys
import csv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import argparse

def del_punc(sentence):
    if sentence.endswith((".", "?", "!")):
        sentence = sentence[:-1]
    return sentence


def promptify_generation_comve(c_statement, f_statement, template_id=0):
    prompt = ''
    if template_id == 0: # baseline, without correct statement
        prompt = f_statement + " This statement is wrong because: "
    elif template_id == 5:
        prompt = "Fact: " + c_statement + "\n Statement: " + f_statement + "\n Based on the difference between the fact and the statement, the statement is wrong because:"
    elif template_id == 1:
        prompt = "Given the fact: " + c_statement + ", explain the following statement based on its difference with the fact: " + f_statement + " The explanation is:"
    elif template_id == 2:
        prompt = "Suppose" + c_statement + ", the following statement is wrong: " + f_statement + " The explanation is:"
    elif template_id == 3:
        prompt = "Given the fact: " + c_statement + ", and the hypothesis: " + f_statement + " The hypothesis is wrong because:"
    elif template_id == 4:
        prompt = "Based on the hint: " + c_statement + ", explain the following statement: " + f_statement + " Let's explain step by step: "
    return prompt


def promptify_generation_esnli(premise, entailment, contradiction, template_id=0):
    prompt = ''
    if template_id == 0: # baseline, without correct statement
        # prompt 0
        # prompt = premise + ". Based on the above description, the following sentence is definitely wrong: " + contradiction + ". The reason is that:"
        # prompt 01
        # prompt = "Based on the fact that " + premise + " It is wrong that " + contradiction + ". The explanation is that:"
        # prompt 02
        # prompt = "Task: Given a sentence, generate an explanation of why this sentence is wrong based on the context.\n" + "Context: " + premise + "\n" +  "Sentence: " + contradiction + "\n"  + "Explanation:"
        # prompt 03
        prompt = "Based on the context that " + premise + ", explain why the following sentence is wrong: " + contradiction + ". The explanation is:"
    elif template_id == 1:
        prompt =  premise + ". Based on the above description, the following sentence is definitely correct: " + entailment + ". However, the next sentence is definitely wrong: " + contradiction + ". The reason is that:"
    elif template_id == 2:
        prompt = "Based on the fact that " + premise + ", it is correct that " + entailment + ". However, it is wrong that " + contradiction + ". The explanation is that:"
    elif template_id == 3:
        prompt = "Based on the fact that " + premise + ", it is correct that " + entailment + " Based on the fact that " + premise +  ". However, it is wrong that " + contradiction + ". The explanation is that:"
    elif template_id == 4:
        prompt = "The context is " + premise + "\n Based on the fact that " + entailment + ", explain why the following sentence is wrong: " + contradiction + ". The explanation is:"
    elif template_id == 5:
        prompt = "Task: Given a sentence, generate an explanation of why this sentence is wrong based on the context and fact.\n" + "Sentence: " + contradiction + "\n" + "Context: " + premise + "\n" + "Fact: " + entailment  + "\n"  + "Explanation:"
    elif template_id == 6:
        prompt = "Task: Based on a context and a related fact, explain why the following sentence is wrong.\n" + "Context: " + premise + "\n" + "Fact: " + entailment  + "\n"  "Sentence: " + contradiction + "\n"  + "Explanation:"
    # print(prompt)
    return prompt

class DataReader:
    def __init__(self, tokenizer, batch_size=1):
        self.tokenizer = tokenizer
        self.data = []
        self.raw_text = []
        self.cnt = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size =  batch_size

    def tokenize(self, text):
        token_id = self.tokenizer.encode(text, add_special_tokens=False)
        bert_tokenized = self.tokenizer.decode(token_id)
        return token_id, bert_tokenized

    def load(self, file_or_example: str, args):
        # if dataloader is not None:
        #     assert isinstance(dataloader, MultiDataLoader) and role is not None
        #     data = dataloader.load_data(role)
        # else:
        with open(file_or_example, 'r', encoding='utf-8') as file:
            lines = list(csv.reader(file))
        self.lines = lines[1:]

        tmp = []
        for line in tqdm(self.lines):
            self.cnt += 1
            if args.task == "comve":
                if args.mode == "top1":
                    c_statement = del_punc(line[-2].strip())
                elif args.mode == "all":
                    c_statement = del_punc(line[-1].strip())
                else:
                    c_statement = del_punc(line[0].strip())
                f_statement = del_punc(line[1].strip())
                tmp.append(promptify_generation_comve(c_statement, f_statement, args.template_id))
            else:
                premise = del_punc(line[0].strip())
                if args.mode == "top1":
                    entailment = del_punc(line[-2].strip())
                elif args.mode == "all":
                    entailment = del_punc(line[-1].strip())
                else:
                    entailment = del_punc(line[1].strip())
                contradiction = del_punc(line[3].strip())
                tmp.append(promptify_generation_esnli(premise, entailment, contradiction, args.template_id))

        self.batched_tokenize(tmp, self.batch_size)

    def batched_tokenize(self, data, batch_size):
        '''
        NOTE: ADD a blank before all texts except for premise!!!
        :param example: {k: v, k: v}
        :return: {k: [{}], 'original_ending': [[{}, {}, {}], []]}
        '''
        # buffer = []
        for i in range(0, len(data), batch_size):
            batch_text = data[i: i+batch_size]
            encoding = self.tokenizer(batch_text, padding=True, return_tensors='pt').to(self.device)
            self.data.append(encoding)
            self.raw_text.append(batch_text)

    def __len__(self):
        return self.cnt


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.plm, torch_dtype=torch.float16).cuda()
    # the fast tokenizer currently does not work correctly
    tokenizer = AutoTokenizer.from_pretrained(args.plm, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    datareader = DataReader(tokenizer, batch_size=args.batch_size)
    datareader.load(args.file, args)

    results = []
    # prompt_length = len(args.prompt)
    for i, encodings in enumerate(datareader.data):
        with torch.no_grad():
            generated_ids = model.generate(**encodings,
                                        pad_token_id=tokenizer.eos_token_id,
                                        max_new_tokens=args.max_length,
                                        do_sample=True,
                                        top_p=0.9)
            text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # print('**********************')
            # print(text) list of [str, str, ...]
            for sample, raw_text in zip(text, datareader.raw_text[i]):
                # print(type(sample), sample) type:str
                # if args.prompt in sample: print('ture')
                idx = len(raw_text)
                # idx = sample.index(args.prompt)
                # print([sample])
                # print([sample[idx+prompt_length:]])
                results.append([sample[idx:]])
            # w.write(text[0].split('\t\t')[-1] + '\n')

    output = json.dumps(results)
    w = open(args.output, 'w')
    w.write(output)

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--plm', '-l', default='./gpt2-xl', help='lm name or local address')
    parser.add_argument('--file', '-f', default="/mnt/lustre/chengsijie/Culprit/data/ComVE/test.csv", help='file to infer')
    parser.add_argument('--output', '-o', default='./results/gpt2-xl.json')
    parser.add_argument('--template_id', '-i', type=int, required=True)
    parser.add_argument('--batch_size', '-b', default=16)
    parser.add_argument('--task', '-t', default="comve")
    parser.add_argument('--mode', '-m', default='gt')
    parser.add_argument('--max_length', '-ml', type=int, default=30)
    args = parser.parse_args()
    print(args)
    main(args)