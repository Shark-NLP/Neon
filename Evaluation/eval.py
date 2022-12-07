#coding='utf-8'

import csv
import logging
import sys
import json
from typing import List, Dict
from itertools import chain

import rouge
import torch
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
import collections
import math
from sentence_transformers import SentenceTransformer, util

EXIT_STATUS_ANSWERS_MALFORMED = 1
EXIT_STATUS_PREDICTIONS_MALFORMED = 2
EXIT_STATUS_PREDICTIONS_EXTRA = 3
EXIT_STATUS_PREDICTION_MISSING = 4


def _clean_text(txt):
    return txt.lower()


def read_pred_file_json(pred_file):
    hypotheses = []
    hypotheses_token = []
    with open(pred_file) as f:
        reader = json.loads(f.read())
        for i, row in enumerate(reader):
            # row = row[0]
            if "###" in row: row = row.split("###")[0]
            tokens = row.split()
            if len(tokens) == 0:
                # print(row, 'empty line found at:', i)
                row = '.'
                # print(word_tokenize(row))
            hypotheses.append(row)
            hypotheses_token.append(word_tokenize(row))
    return hypotheses, hypotheses_token


def read_pred_file(pred_file):
    hypotheses = []
    hypotheses_token = []
    with open(pred_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            hypotheses.append(row[0])
            hypotheses_token.append(word_tokenize(row[0]))
    return hypotheses, hypotheses_token


def read_gold_file(gold_file, args):
    references = []
    references_token = []
    with open(gold_file) as csvfile:
        reader = list(csv.reader(csvfile))
        for row in reader[1:]:
            if args.task == "comve":
                references.append([row[2], row[5], row[6]])
                references_token.append([word_tokenize(row[2]), word_tokenize(row[5]), word_tokenize(row[6])])
            elif args.task == "esnli":
                references.append([row[4], row[5], row[6]])
                references_token.append([word_tokenize(row[4]), word_tokenize(row[5]), word_tokenize(row[6])])
    return references, references_token


def process_data(pred_file, gold_file, args):
    hypotheses, hypotheses_token = read_pred_file_json(pred_file)
    references, references_token = read_gold_file(gold_file, args)
    return hypotheses, hypotheses_token, references, references_token


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def calculate_bleu(references: Dict[str, List[List[str]]],
                   predictions: Dict[str, List[str]],
                   max_order=4,
                   smooth=False) -> float:

    reference_corpus = []
    prediction_corpus = []

    for instance_id, reference_sents in references.items():
        try:
            prediction_sent = predictions[instance_id]
        except KeyError:
            logging.error("Missing prediction for instance '%s'.", instance_id)
            sys.exit(EXIT_STATUS_PREDICTION_MISSING)

        del predictions[instance_id]

        prediction_corpus.append(prediction_sent)
        reference_corpus.append(reference_sents)

    if len(predictions) > 0:
        logging.error("Found %d extra predictions, for example: %s", len(predictions),
                      ", ".join(list(predictions.keys())[:3]))
        sys.exit(EXIT_STATUS_PREDICTIONS_EXTRA)

    score = _compute_bleu(reference_corpus, prediction_corpus,
                          max_order=max_order, smooth=smooth)[0]

    return score


def eval_rouge(pred_endings_file, gold_file, args):
    ori_hypotheses, _, ori_references, _ = process_data(pred_endings_file, gold_file, args)

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    references = []
    hypotheses = []
    for r, h in zip(ori_references, ori_hypotheses):
        clean_reference = [_clean_text(i) for i in r]
        clean_hypothesis = _clean_text(h)
        if len(clean_hypothesis) == 0:
            assert False
        references.append(clean_reference)
        hypotheses.append(clean_hypothesis)
        # print(hypotheses)
        # exit()
    assert len(references) == len(hypotheses)
    scores = evaluator.get_scores(hypotheses, references)
    return {'rouge_all': scores}


def eval_bert_score(pred_endings_file, gold_file, args, bert_model="bert-base-uncased"):
    ori_hypotheses, _, ori_references, _ = process_data(pred_endings_file, gold_file, args)

    references = []
    hypotheses = []
    for r, h in zip(ori_references, ori_hypotheses):
        clean_reference = [_clean_text(i) for i in r]
        clean_hypothesis = _clean_text(h)
        if len(clean_hypothesis) == 0:
            assert False
        hypotheses.append(clean_hypothesis)
        references.append(clean_reference)

    assert len(references) == len(hypotheses)
    P, R, F1 = bert_score(hypotheses, references, model_type=bert_model,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    return {
        "bert_score_P": P.mean().item(),
        "bert_score_R": R.mean().item(),
        "bert_score_F1": F1.mean().item(),
        # "bert_score_P_by_instance": [float(f) for f in list(P.numpy())],
        # "bert_score_R_by_instance": [float(f) for f in list(R.numpy())],
        # "bert_score_F1_by_instance": [float(f) for f in list(F1.numpy())],
    }


def eval_sent_bert(pred_endings_file, gold_file, args):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    ori_hypotheses, _, ori_references, _ = process_data(pred_endings_file, gold_file, args)

    references = []
    hypotheses = []
    for r, h in zip(ori_references, ori_hypotheses):
        for i in r:
            clean_reference = _clean_text(i)
            clean_hypothesis = _clean_text(h)
            if len(clean_hypothesis) == 0:
                assert False
            references.append(clean_reference)
            hypotheses.append(clean_hypothesis)

    # Compute embedding for both lists
    embeddings1 = sbert_model.encode(hypotheses, convert_to_tensor=True)
    embeddings2 = sbert_model.encode(references, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    sent_bert_score = 0
    cnt = 0
    for i in range(len(ori_hypotheses)):
        sent_bert_score += max(cosine_scores[i*3][i*3], cosine_scores[i*3+1][i*3+1], cosine_scores[i*3+2][i*3+2])
        # sent_bert_score += cosine_scores[i][i]
    sent_bert_score /= len(ori_hypotheses)

    return {'sent-bert': sent_bert_score}


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def distinct_n_sentence_level(sent, n):
    distinct_ngrams = set(ngrams(sent, n))
    return len(distinct_ngrams) / len(sent)


def eval_DIST(pred_endings_file, gold_file):
    ori_hypotheses, _, _, _ = process_data(pred_endings_file, gold_file, args)

    hypotheses = []
    for h in ori_hypotheses:
        clean_hypothesis = _clean_text(h)
        if len(clean_hypothesis) == 0:
            assert False
        hyp_tokens = clean_hypothesis.strip().split(" ")
        hypotheses.append(hyp_tokens)
        # print(hypotheses)
        # exit()

    score_1 = sum(distinct_n_sentence_level(hypothese, 1) for hypothese in hypotheses) / len(hypotheses)
    score_2 = sum(distinct_n_sentence_level(hypothese, 2) for hypothese in hypotheses) / len(hypotheses)

    return {
        "dist_1": score_1,
        "dist_2": score_2
    }


def read_references(filename: str, args) -> List[List[List[str]]]:
    references = {}
    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            for i, row in enumerate(list(reader)[1:]):
                instance_id = i
                if args.task == "comve":
                    references_raw1 = row[2]
                    references_raw2 = row[5]
                    references_raw3 = row[6]
                elif args.task == "esnli":
                    references_raw1 = row[4]
                    references_raw2 = row[5]
                    references_raw3 = row[6]
                tokens = []
                for ref in [references_raw1, references_raw2, references_raw3]:
                    if ref:
                        tokens.append(ref.split())

                if len(tokens) == 0:
                    logging.error(
                        "No reference sentence in file %s on line %d", filename, reader.line_num)
                    sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

                references[instance_id] = tokens

        except csv.Error as e:
            logging.error('file %s, line %d: %s', filename, reader.line_num, e)
            sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

    return references


def read_predictions(filename: str) -> List[List[str]]:
    predictions = {}
    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            for i, row in enumerate(reader):
                instance_id = i
                prediction_raw = row[0]
                tokens = prediction_raw.split()
                predictions[instance_id] = tokens
        except csv.Error as e:
            logging.error('file %s, line %d: %s', filename, reader.line_num, e)
            sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

    return predictions


def read_predictions_json(filename: str) -> List[List[str]]:
    predictions = {}
    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = json.loads(f.read())
        for i, row in enumerate(reader):
            # row = row[0]
            instance_id = i
            if "###" in row: row = row.split("###")[0]
            # prediction_raw = row[1]
            tokens = row.split()
            if len(tokens) == 0: tokens = ['.']
            predictions[instance_id] = tokens

    return predictions


def main():
    # hypotheses, references = process_data(args.pred_file, args.gold_file)

    # csvwriter = open("./results/dev_eval_bleu.csv", 'w', newline='')
    # csv_writer = csv.writer(csvwriter)

    results = {}
    # BLEU
    references = read_references(args.gold_file, args)
    predictions = read_predictions_json(args.pred_file)
    bleu = calculate_bleu(references, predictions,
                          max_order=args.max_order, smooth=args.smooth)
    results.update({"bleu_score": bleu})
    # # # bleu = eval_bleu(args.pred_file, args.gold_file)
    # ROUGE
    results.update(eval_rouge(args.pred_file, args.gold_file, args))
    rscore = results.pop('rouge_all')
    results['rouge-l'] = rscore['rouge-l']
    # BERTScore
    results.update(eval_bert_score(args.pred_file, args.gold_file, args))
    # S-BERT
    results.update(eval_sent_bert(args.pred_file, args.gold_file, args))
    # DIST-N
    results.update(eval_DIST(args.pred_file, args.gold_file))
    print(results)
    for k,v in results.items():
        if k == 'bleu_score': print(k, round(v, 4))
        elif k == "rouge-l": print(k, round(v['f'], 4))
        elif k == "bert_score_F1": print(k, round(v, 4))
        elif k == "sent-bert": print(k, v)
        # elif 'dist' in k: print(k, round(v, 4))
    print('**'*20+args.pred_file+'**'*20)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', '-p', help='predict csv file name. '
                                               'e.g python3 metric.py -p pre_file.csv -i gold_file.csv')
    parser.add_argument('--gold_file', '-g', help='ground_truth csv file name. '
                                              'e.g python3 metric.py -p pre_file.csv -i gold_file.csv')
    parser.add_argument(
        '--max_order', default=4, type=int, help='Maximum n-gram order to use when computing BLEU score')
    parser.add_argument('--smooth', action='store_true',
                        help='Whether or not to apply Lin et al. 2004 smoothing')
    parser.add_argument('--task', '-t', type=str, default="comve", help='[comve, esnli]')
    args = parser.parse_args()
    main()