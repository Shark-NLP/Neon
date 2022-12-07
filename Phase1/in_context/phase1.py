import argparse
import requests
import json
import csv
import random
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

def sample(reader):
    instances = random.sample(reader, 16)
    return instances


def del_punc(sentence):
    if sentence.endswith((".", "?", "!")):
        sentence = sentence[:-1]
    return sentence


def construct_prompt_ComVE(instances):
    prompt = "Task: Based on the incorrect statement, generate the correct statement.\n"
    for instance in instances:
        sent = "Incorrect statement: " + del_punc(instance[1]) + ".\n" + "Correct statement:\n" + del_punc(instance[0]) + ".\n" + "###\n"
        prompt += sent
    return prompt


def construct_prompt_esnli(instances):
    prompt = "Task: Given the premise and the incorrect statement, generate the correct statement.\n"
    for instance in instances:
        premise = del_punc(instance[0].strip())
        contradiction = del_punc(instance[3].strip())
        entailment = del_punc(instance[1].strip())
        sent = "Premise: " + premise + ".\n" + "Incorrect statement: " + contradiction + ".\n" + "Correct statement:\n" + entailment + ".\n" + "###\n"
        prompt += sent
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_file", default="sampled_instance.csv", type=str, required=False,
                        help="the file of sampled instances")
    parser.add_argument("--test_file", default="../data/ComVE/test.csv", type=str, required=False, help="the file of test dataset")
    parser.add_argument("--task", default="comve", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_tokens", default="25", type=str, required=True)
    args = parser.parse_args()

    random.seed(7)

    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    url = ""

    fw_ours = open(args.output_file, "w", encoding="utf-8")
    ours_writer = csv.writer(fw_ours)

    sample_fr = open(file, "r", encoding="utf-8")
    sample_reader = list(csv.reader(sample_fr))[1:]

    queries = []
    logger.info("start in-context learning.........")
    with open(args.test_file, "r", encoding="utf-8") as fr:
        reader = list(csv.reader(fr))[1:]
        cnt = 0
        for line in reader:
            cnt += 1
            logger.info("generate the " + str(cnt) + "-th correct sentence")
            sampled_ins = sample(sample_reader)
            if args.task == "comve":
                c_statement = del_punc(line[0])
                f_statement = del_punc(line[1])
                prompt = construct_prompt_ComVE(sampled_ins)
                prompt += "Incorrect statement: " + f_statement + ".\n" + "Correct statement:\n"
                queries.append(prompt)
                # ours_result = {"prompt": prompt, "max_tokens": "25", "top_p": 0.9}
            elif args.task == "esnli":
                premise = del_punc(line[0].strip())
                contradiction = del_punc(line[3].strip())
                prompt = construct_prompt_esnli(sampled_ins)
                prompt += "Premise: " + premise + ".\n" + "Incorrect statement: " + contradiction + ".\n" + "Correct statement:\n"
                queries.append(prompt)
                # ours_result = {"prompt": queries, "max_tokens": "40", "top_p": 0.9}
    ours_result = {"prompt": queries, "max_tokens": args.max_tokens, "top_p": 0.9}
    logger.info("generating......")
    response = json.loads(requests.post(url, data=json.dumps(ours_result), headers=headers).text)
    res = [r['text'].encode("utf-8").decode("unicode_escape") for r in response['choices']]
    # print(res[0].split("###")[0].strip())
    ours_writer.writerow([res[0].split("###")[0].strip()])
    logger.info("over.......")

    fw_ours.close()