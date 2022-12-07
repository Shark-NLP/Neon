#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import requests
import csv

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
        prompt =  "Given the fact: " + c_statement + ", explain the following statement based on its difference with the fact: " + f_statement + " The explanation is:"
    elif template_id == 2:
        prompt =  "Suppose" + c_statement + ", the following statement is wrong: " + f_statement + " The explanation is:"
    elif template_id == 3:
        prompt =  "Given the fact: " + c_statement + ", and the hypothesis: " + f_statement + ". The hypothesis is wrong because:"
    elif template_id == 4:
        prompt =  "Based on the hint: " + c_statement + ", explain the following statement: " + f_statement + " Let's explain step by step: "
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--template_id', type=int, default=3)
parser.add_argument('--output_file', '-o', required=True)
parser.add_argument('--input_file', '-i', default='test_all.csv')
parser.add_argument('--num', '-n', type=int, default=1)
parser.add_argument('--task', '-t', default="comve")
args = parser.parse_args()
headers = {
    "Content-Type": "application/json; charset=UTF-8"
    }
url = ""
with open(args.input_file, 'r', encoding='utf-8') as file:
    lines = list(csv.reader(file))
lines = lines[1:]

data = []
for i, line in enumerate(lines):
    if args.task == "comve":
        if args.num >= 1:
            c_statement = "1. " + del_punc(line[0].strip())
        if args.num >= 2:
            c_statement += " 2. " + del_punc(line[-5].strip())
        if args.num >= 3:
            c_statement += " 3. " + del_punc(line[-4].strip())
        if args.num >= 4:
            c_statement += " 4. " + del_punc(line[-3].strip())
        if args.num >= 5:
            c_statement += " 5. " + del_punc(line[-2].strip())
        f_statement = del_punc(line[1].strip())
        prompt_text = promptify_generation_comve(c_statement, f_statement, template_id=args.template_id)
    elif args.task == "esnli":
        premise = del_punc(line[0].strip())
        if args.num >= 1:
            entailment = "1. " + del_punc(line[1].strip())
        if args.num >= 2:
            entailment += " 2. " + del_punc(line[-5].strip())
        if args.num >= 3:
            entailment += " 3. " + del_punc(line[-4].strip())
        if args.num >= 4:
            entailment += " 4. " + del_punc(line[-3].strip())
        if args.num >= 5:
            entailment += " 5. " + del_punc(line[-2].strip())
        contradiction = del_punc(line[3].strip())
        prompt_text = promptify_generation_esnli(premise, entailment, contradiction, args.template_id)

    data.append(prompt_text)

pyload = {"prompt": data, "max_tokens": "30", "top_p": 0.9, "seed": 1024}
response = json.loads(requests.post(url, data=json.dumps(pyload), headers=headers).text)
res = [r['text'] for r in response['choices']]
with open(args.output_file, 'w') as json_file:
    json.dump(res, json_file)