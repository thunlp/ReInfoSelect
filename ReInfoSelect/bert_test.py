import argparse
import pickle
import csv
import os
import re
import json

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_transformers import *

import pytrec_eval

epoch = 1
debug = False
task = 'robust'

eval_batch_size = 32
stem = 150

fold = '1'
#test_file = task + '/cross_validate/dev_09_' + fold + '.txt'
#test_file = task + '/cross_validate/dev.txt'
test_file = "/home3/zhangkaitao/retrieval/cx_test.tsv"
#pretrained_model = '/home1/zhangkaitao/bert-base/bert_base'# + fold
#pretrained_model = '../models/multi_trans_c100_new_bert_rl_pp_rb'# + fold
pretrained_model = '../models/reinfose_bert_cx'
#out_file = 'util/' + task + '/dev_09_' + fold + '.txt'
#out_file = 'util/' + task + '/dev_09.txt'

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

#with open(task + '/qrels', 'r') as f_qrel:
with open('/home3/zhangkaitao/retrieval/test_cx/qrels', 'r') as f_qrel:
    qrel = pytrec_eval.parse_qrel(f_qrel)

def raw2tok(s):
    lst = regex_multi_space.sub(' ', regex_drop_char.sub(' ', s.lower())).strip().split()
    return lst

class devFeatures(object):
    def __init__(self, query_id, doc_id, qd_score, query, doc, raw_score, d_input_ids, d_input_mask, d_segment_ids):
        self.query_id = query_id
        self.doc_id = doc_id
        self.qd_score = qd_score
        self.query = query
        self.doc = doc
        self.raw_score = raw_score
        self.d_input_ids = d_input_ids
        self.d_input_mask = d_input_mask
        self.d_segment_ids = d_segment_ids

def pack_bert_seq(q_tokens, p_tokens, tokenizer, max_seq_length):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in q_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in p_tokens:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids

def read_clueweb_to_features(input_file, tokenizer, is_training=True):
    max_seq_length = 256
    max_query_length = 32
    with open(input_file, 'r') as reader:
        cnt = 0
        features = []
        for line in reader:
            cnt += 1
            if debug and len(features) >= 100:# debug
                break
            s = line.strip('\n').split('\t')

            #query_toks_raw = raw2tok(s[0])
            #doc_toks_raw = raw2tok(s[1])

            qd_score = 0
            query_id = s[3]
            doc_id = s[4]
            raw_score = 0.0

            #s[0] = ' '.join(query_toks_raw)
            #s[1] = ' '.join(doc_toks_raw[:stem])

            q_tokens = tokenizer.tokenize(s[0])
            if len(q_tokens) > max_query_length:
                q_tokens = q_tokens[:max_query_length]

            max_doc_length = max_seq_length - len(q_tokens) - 3
            d_tokens = tokenizer.tokenize(s[1])
            if len(d_tokens) > max_doc_length:
                d_tokens = d_tokens[:max_doc_length]

            d_input_ids, d_input_mask, d_segment_ids = pack_bert_seq(q_tokens, d_tokens, tokenizer, max_seq_length)

            features.append(devFeatures(
                query_id = query_id,
                doc_id = doc_id,
                qd_score = qd_score,
                query = s[0],
                doc = s[1],
                raw_score = raw_score,
                d_input_ids = d_input_ids,
                d_input_mask = d_input_mask,
                d_segment_ids = d_segment_ids))

        return features

def devDataLoader(features, batch_size):
    n_samples = len(features)
    idx = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = idx[start_idx:start_idx+batch_size]

        query_id = [features[i].query_id for i in batch_idx]
        doc_id = [features[i].doc_id for i in batch_idx]
        qd_score = [features[i].qd_score for i in batch_idx]
        query = [features[i].query for i in batch_idx]
        doc = [features[i].doc for i in batch_idx]
        raw_score = [features[i].raw_score for i in batch_idx]
        d_input_ids = torch.tensor([features[i].d_input_ids for i in batch_idx], dtype=torch.long)
        d_input_mask = torch.tensor([features[i].d_input_mask for i in batch_idx], dtype=torch.long)
        d_segment_ids = torch.tensor([features[i].d_segment_ids for i in batch_idx], dtype=torch.long)

        batch = (query_id, doc_id, qd_score, query, doc, raw_score, d_input_ids, d_input_mask, d_segment_ids)
        yield batch
    return

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set_seed()

    config = BertConfig.from_pretrained('/home1/zhangkaitao/bert-base/bert_config.json')
    tokenizer = BertTokenizer.from_pretrained('/home1/zhangkaitao/bert-base/bert-base-uncased-vocab.txt')
    #config = BertConfig.from_pretrained('/home3/zhangkaitao/nyu_msmarco/models/bert/bert_config_l.json')
    #tokenizer = BertTokenizer.from_pretrained('/home3/zhangkaitao/nyu_msmarco/models/bert/vocab.txt')
    model = BertForRanking.from_pretrained('/home1/zhangkaitao/bert-base/pytorch_model.bin', config=config)
    model.train()

    state_dict=torch.load(pretrained_model)
    model.load_state_dict(state_dict)

    model.to(device)

    # test data
    test_features = read_clueweb_to_features(test_file, tokenizer, is_training=False)
    test_data = devDataLoader(test_features, eval_batch_size)

    # test
    rst_dict = {}
    for s, batch in enumerate(test_data):
        query_id = batch[0]
        doc_id = batch[1]
        qd_score = batch[2]
        query = batch[3]
        doc = batch[4]
        raw_score = batch[5]
        batch = tuple(t.to(device) for t in batch[6:])
        (d_input_ids, d_input_mask, d_segment_ids) = batch

        with torch.no_grad():
            doc_scores, doc_features = model(d_input_ids, d_segment_ids, d_input_mask)
        d_scores = doc_scores.detach().cpu().tolist()
        d_features = doc_features.detach().cpu().tolist()

        for (q_id, d_id, qd_s, q, d, r_s, d_s, d_f) in zip(query_id, doc_id, qd_score, query, doc, raw_score, d_scores, d_features):
            line = []
            line.append(str(qd_s))
            line.append('id:' + q_id)
            for i, fi in enumerate(d_f):
                line.append(str(i+1) + ':' + str(fi))
            if task != 'MQ2007':
                line.append(str(i+2) + ':' + str(r_s))# toby change
            if q_id in rst_dict:
                rst_dict[q_id].append((qd_s, d_s, d_id, q, d))
            else:
                rst_dict[q_id] = [(qd_s, d_s, d_id, q, d)]

    out_trec = 'rebert.txt'
    with open(out_trec, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores, key=lambda x: x[1], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id+' '+'Q0'+' '+str(value[2])+' '+str(rank+1)+' '+str(value[1])+' '+'Conv-KNRM'+'\n')

    '''
    out_trec = 'rebert.json'
    with open(out_trec, 'w') as writer:
        tmp = {"query_id": "", "records": []}
        for q_id, scores in rst_dict.items():
            tmp["query_id"] = q_id
            tmp["records"] = []
            max_pool = []
            res = sorted(scores, key=lambda x: x[1], reverse=True)
            for rank, value in enumerate(res):
                if value[2] not in max_pool:
                    max_pool.append(value[2])
                    tmp["records"].append({"paper_id":value[2], "score":value[1], "paragraph":value[4]})
                    #writer.write(q_id+'\t'+str(value[2])+'\t'+str(rank+1)+'\n')
            writer.write(json.dumps(tmp) + '\n')
    '''

if __name__ == "__main__":
    main()
