#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import requests
import csv
import re


def del_punc(sentence):
    if sentence.endswith((".", "?", "!")):
        sentence = sentence[:-1] + "."
    return sentence


def process(sentence):
    sentence = re.sub('[\d+.]', '', sentence)
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

def promptify_generation_esnli_entailment(premise, facts, entailment, template_id=0):
    prompt = ''
    if template_id == 0: # baseline, without correct statement
        # prompt 0
        # prompt = premise + ". Based on the above description, the following sentence is definitely wrong: " + contradiction + ". The reason is that:"
        # prompt 01
        # prompt = "Based on the fact that " + premise + " It is wrong that " + contradiction + ". The explanation is that:"
        # prompt 02
        # prompt = "Task: Given a sentence, generate an explanation of why this sentence is wrong based on the context.\n" + "Context: " + premise + "\n" +  "Sentence: " + contradiction + "\n"  + "Explanation:"
        # prompt 03
        prompt = "Based on the context that " + premise + ", explain why the following sentence is correct: " + entailment + ". The explanation is:"
    elif template_id == 1:
        prompt = premise + ". Based on the above description, the following sentence is definitely correct: " + facts + ".The next sentence is also correct: " + entailment + ". The reason is that:"
    elif template_id == 2:
        prompt = "Based on the fact that " + premise + ", it is correct that " + facts + ". It is also correct " + entailment + ". The explanation is that:"
    elif template_id == 3:
        prompt = "Based on the fact that " + premise + ", it is correct that " + facts + " Based on the fact that " + premise +  ". it is also correct " + entailment + ". The explanation is that:"
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
        prompt = premise + ". Based on the above description, the following sentence is definitely correct: " + entailment + ". However, the next sentence is definitely wrong: " + contradiction + ". The reason is that:"
    elif template_id == 2:
        prompt = "Based on the fact that " + premise + ", it is correct that " + entailment + ". However, it is wrong that " + contradiction + ". The explanation is that:"
    elif template_id == 3:
        prompt = "Based on the fact that " + premise + ", it is correct that " + entailment + " Based on the fact that " + premise +  ". However, it is wrong that " + contradiction + ". The explanation is that:"
    elif template_id == 4:
        prompt = "The context is " + premise + "\n Based on the fact that " + entailment + ", explain why the following sentence is wrong: " + contradiction + ". The explanation is:"
    elif template_id == 5:
        prompt = "Based on the premise that " + premise + ", and the facts " + entailment + ", explain why the following statement is wrong: " + contradiction + ". Because"
    elif template_id == 6:
        prompt = "Given the premise that " + premise + ", and the facts " + entailment + ", explain why the premise is contradiction with the following statement: " + contradiction + ".Because"
    # elif template_id == 5:
    #     prompt = "Task: Given a sentence, generate an explanation of why this sentence is wrong based on the context and fact.\n" + "Sentence: " + contradiction + "\n" + "Context: " + premise + "\n" + "Fact: " + entailment  + "\n"  + "Explanation:"
    # elif template_id == 6:
    #     prompt = "Task: Based on a context and a related fact, explain why the following sentence is wrong.\n" + "Context: " + premise + "\n" + "Fact: " + entailment  + "\n"  "Sentence: " + contradiction + "\n"  + "Explanation:"
    # print(prompt)
    return prompt


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--template_id', '-p', type=int, default=3)
parser.add_argument('--output_file', '-o', required=True)
parser.add_argument('--input_file', '-i', default='test_all.csv')
parser.add_argument('--task', '-t', default="comve")
parser.add_argument('--mode', '-m', default="top1")
parser.add_argument('--seed', '-s', type=int, default=1024)
args = parser.parse_args()
headers = {
    "Content-Type": "application/json; charset=UTF-8"
    }
url = "http://10.140.1.39:6010/completions"
with open(args.input_file, 'r', encoding='utf-8') as file:
    lines = list(csv.reader(file))
lines = lines[1:]

data = []
for i, line in enumerate(lines):
    if args.task == "comve":
        if args.mode == "top1":
            c_statement = del_punc(line[-2].strip())
        # elif args.mode == "filter":
        #     c_statement = process(del_punc(line[-1].strip()))
        elif args.mode == "all":
            c_statement = process(del_punc(line[-1].strip()))
        f_statement = del_punc(line[1].strip())
        prompt_text = promptify_generation_comve(c_statement, f_statement, template_id=args.template_id)
    elif args.task == "esnli":
        if args.mode == "top1":
            entailment = process(del_punc(line[-2].strip()))
        elif args.mode == "all":
            entailment = process(del_punc(line[-1].strip()))
        premise = del_punc(line[0].strip())
        contradiction = del_punc(line[3].strip())
        prompt_text = promptify_generation_esnli(premise, entailment, contradiction, args.template_id)
    elif args.task == "entail":
        if args.mode == "top1":
            facts = del_punc(line[-2].strip())
        elif args.mode == "all":
            facts = del_punc(line[-1].strip())
        premise = del_punc(line[0].strip())
        entailment = del_punc(line[1].strip())
        prompt_text = promptify_generation_esnli_entailment(premise, facts, entailment, args.template_id)

    data.append(prompt_text)

pyload = {"prompt": data, "max_tokens": "30", "top_p": 0.9, "seed": args.seed}
response = json.loads(requests.post(url, data=json.dumps(pyload), headers=headers).text)
res = [r['text'] for r in response['choices']]
with open(args.output_file, 'w') as json_file:
    json.dump(res, json_file)