import re
import json

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from nltk.corpus import stopwords
sws = {}
for w in stopwords.words('english'):
    sws[w] = 1
from MyRanker_feature import cknrm
from krovetzstemmer import Stemmer
stemmer = Stemmer()

import pytrec_eval

epoch = 1
eval_batch_size = 32
stem = 128

test_file = "/home3/zhangkaitao/retrieval/cx_test.tsv"
pretrained_model = '../models/reinfoselect_cknrm_yj'
with open('/home3/zhangkaitao/retrieval/test_cx/cc_qrels.txt', 'r') as f_qrel:
    qrel = pytrec_eval.parse_qrel(f_qrel)

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

def raw2tok(s):
    lst = regex_multi_space.sub(' ', regex_drop_char.sub(' ', s.lower())).strip().split()
    return lst

def load_glove(glove_file):
    idx = 0
    idx2word = []
    word2idx = {}
    word2vec = {}

    # process unk pad
    idx2word.append('<PAD>')
    word2idx['<PAD>'] = idx
    word2vec['<PAD>'] = np.random.normal(scale=0.6, size=(300, ))
    idx += 1

    idx2word.append('<UNK>')
    word2idx['<UNK>'] = idx
    word2vec['<UNK>'] = np.random.normal(scale=0.6, size=(300, ))
    idx += 1

    with open(glove_file, 'r') as f:
        for line in f:
            val = line.split()
            idx2word.append(val[0])
            word2idx[val[0]] = idx
            word2vec[val[0]] = np.asarray(val[1:], dtype='float32')
            idx += 1

    return idx2word, word2idx, word2vec

def create_embeddings(idx2word, word2vec):
    embedding_matrix = np.zeros((len(idx2word), 300))
    for idx, word in enumerate(idx2word):
        embedding_matrix[idx] = word2vec[word]

    return embedding_matrix

class devFeatures(object):
    def __init__(self, query_id, doc_id, qd_score, query, doc, raw_score, query_idx, doc_idx, query_len, doc_len):
        self.query_id = query_id
        self.doc_id = doc_id
        self.qd_score = qd_score
        self.query = query
        self.doc = doc
        self.raw_score = raw_score
        self.query_idx = query_idx
        self.doc_idx = doc_idx
        self.query_len = query_len
        self.doc_len = doc_len

def tok2idx(toks, word2idx):
    input_ids = []
    for tok in toks:
        if tok in word2idx:
            input_ids.append(word2idx[tok])
        else:
            input_ids.append(word2idx['<UNK>'])
    return input_ids

def filter_sw(toks, stem):
    wordsFiltered = []
    for w in toks:
        if w not in sws:
            w = stemmer.stem(w)
            if len(wordsFiltered) >= stem:
                break
            wordsFiltered.append(w)
    return wordsFiltered

def read_clueweb_to_features(input_file, word2idx, is_training=True):
    with open(input_file, 'r') as reader:
        cnt = 0
        features = []
        for line in reader:
            cnt += 1
            s = line.strip('\n').split('\t')

            qd_score = 0
            query_id = s[3]
            doc_id = s[4]
            raw_score = float(s[5])
            query_toks = filter_sw(raw2tok(s[0]), 20)
            doc_toks = filter_sw(raw2tok(s[1]), stem)

            query_len = len(query_toks)
            doc_len = len(doc_toks)

            while len(query_toks) < 3:
                query_toks.append('<PAD>')
            while len(doc_toks) < 3:
                doc_toks.append('<PAD>')

            query_idx = tok2idx(query_toks, word2idx)
            doc_idx = tok2idx(doc_toks, word2idx)

            features.append(devFeatures(
                query_id = query_id,
                doc_id = doc_id,
                qd_score = qd_score,
                query = s[0],
                doc = s[1],
                raw_score = raw_score,
                query_idx = query_idx,
                doc_idx = doc_idx,
                query_len = query_len,
                doc_len = doc_len))

        return features

class Multi_Trans(nn.Module):
    def __init__(self, embedding_matrix):
        super(Multi_Trans, self).__init__()
        em = torch.tensor(embedding_matrix, dtype=torch.float32).cuda()
        self.embedding = nn.Embedding(400002, 300)
        self.embedding.weight = nn.Parameter(em)
        self.embedding.weight.requires_grad = True
        self.ranker = cknrm(21, 300, True, True)
    
    def forward(self, query_idx, pos_idx, neg_idx, query_len, pos_len, neg_len, raw_score=None, is_training=True):
        query_embed = self.embedding(query_idx)
        doc_embed = self.embedding(pos_idx)
        query_mask = self.create_mask_like(query_len, query_embed)
        doc_mask = self.create_mask_like(pos_len, doc_embed)

        doc_scores, doc_features = self.ranker(query_embed, doc_embed, query_mask, doc_mask)#, raw_score)

        return doc_scores, doc_features
    
    def create_mask_like(self, lengths, like):
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        return mask

def devDataLoader(features, batch_size):
    batches = []
    n_samples = len(features)
    idx = np.arange(n_samples)
    for start_idx in range(0, n_samples, batch_size):
        batch_idx = idx[start_idx:start_idx+batch_size]

        query_id = [features[i].query_id for i in batch_idx]
        doc_id = [features[i].doc_id for i in batch_idx]
        qd_score = [features[i].qd_score for i in batch_idx]
        query = [features[i].query for i in batch_idx]
        doc = [features[i].doc for i in batch_idx]
        raw_score = torch.tensor([features[i].raw_score for i in batch_idx], dtype=torch.float)
        query_idx = [torch.tensor(features[i].query_idx, dtype=torch.long) for i in batch_idx]
        doc_idx = [torch.tensor(features[i].doc_idx, dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i].query_len for i in batch_idx], dtype=torch.long)
        doc_len = torch.tensor([features[i].doc_len for i in batch_idx], dtype=torch.long)

        query_idx = nn.utils.rnn.pad_sequence(query_idx, batch_first=True)
        doc_idx = nn.utils.rnn.pad_sequence(doc_idx, batch_first=True)

        batch = (query_id, doc_id, qd_score, query, doc, raw_score, query_idx, doc_idx, query_len, doc_len)
        batches.append(batch)
    return batches

def main():
    idx2word, word2idx, word2vec = load_glove('data/glove.6B.300d.txt')
    embedding_matrix = create_embeddings(idx2word, word2vec)

    model = Multi_Trans(embedding_matrix)
    state_dict=torch.load(pretrained_model)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # test data
    test_features = read_clueweb_to_features(test_file, word2idx, is_training=False)
    test_data = devDataLoader(test_features, eval_batch_size)

    # test
    rst_dict = {}
    all_time = 0
    fout = open('recknrm_features', 'w')
    for s, batch in enumerate(test_data):
        query_id = batch[0]
        doc_id = batch[1]
        qd_score = batch[2]
        query = batch[3]
        doc = batch[4]
        batch = tuple(t.to(device) for t in batch[5:])
        (raw_score, query_idx, doc_idx, query_len, doc_len) = batch

        with torch.no_grad():
            doc_scores, doc_features = model(query_idx, doc_idx, None, query_len, doc_len, None, raw_score, False)
        d_scores = doc_scores.detach().cpu().tolist()
        d_features = doc_features.detach().cpu().tolist()
        raw_score = raw_score.detach().cpu().tolist()
        for (q_id, d_id, qd_s, q, d, r_s, d_s, d_f) in zip(query_id, doc_id, qd_score, query, doc, raw_score, d_scores, d_features):
            line = []
            line.append(str(qd_s))
            line.append('id:' + q_id)
            line.append(str(1) + ':' + str(d_s))
            line.append('#' + d_id)
            fout.write(' '.join(line) + '\n')
            if q_id in rst_dict:
                rst_dict[q_id].append((qd_s, d_s, d_id, q, d))
            else:
                rst_dict[q_id] = [(qd_s, d_s, d_id, q, d)]
    fout.close()

    out_trec = 'recknrm.jsonl'
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
            writer.write(json.dumps(tmp) + '\n')

if __name__ == "__main__":
    main()
