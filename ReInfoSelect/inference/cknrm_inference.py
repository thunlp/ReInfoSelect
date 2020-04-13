import re
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from CKNRM import CKNRM

from nltk.corpus import stopwords
sws = {}
for w in stopwords.words('english'):
    sws[w] = 1
from krovetzstemmer import Stemmer
stemmer = Stemmer()

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
    def __init__(self, query_id, doc_id, query, doc, raw_score, query_idx, doc_idx, query_len, doc_len):
        self.query_id = query_id
        self.doc_id = doc_id
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

def filter_sw(toks, length):
    wordsFiltered = []
    for w in toks:
        if w not in sws:
            w = stemmer.stem(w)
            if len(wordsFiltered) >= length:
                break
            wordsFiltered.append(w)
    return wordsFiltered

def read_data_to_features(input_file, word2idx, args):
    with open(input_file, 'r') as reader:
        cnt = 0
        features = []
        for line in reader:
            cnt += 1
            s = json.loads(line)

            query_id = s['query_id']
            query_toks = filter_sw(raw2tok(s['query']), args.max_query_len)
            query_len = len(query_toks)
            while len(query_toks) < 3:
                query_toks.append('<PAD>')
            query_idx = tok2idx(query_toks, word2idx)

            for rec in s['records']:
                doc_id = rec['paper_id']
                raw_score = float(rec['score'])
                doc_toks = filter_sw(raw2tok(rec['paragraph']), args.max_doc_len)
                doc_len = len(doc_toks)
                while len(doc_toks) < 3:
                    doc_toks.append('<PAD>')
                doc_idx = tok2idx(doc_toks, word2idx)

                features.append(devFeatures(
                    query_id = query_id,
                    doc_id = doc_id,
                    query = s['query'],
                    doc = rec['paragraph'],
                    raw_score = raw_score,
                    query_idx = query_idx,
                    doc_idx = doc_idx,
                    query_len = query_len,
                    doc_len = doc_len))

        return features

class Ranker(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(Ranker, self).__init__()
        em = torch.tensor(embedding_matrix, dtype=torch.float32).cuda()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.embedding.weight = nn.Parameter(em)
        self.embedding.weight.requires_grad = True
        self.ranker = CKNRM(args.kernel_size, args.embedding_dim, args.cnn_kernel, args.cuda, True)
    
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
        query = [features[i].query for i in batch_idx]
        doc = [features[i].doc for i in batch_idx]
        raw_score = torch.tensor([features[i].raw_score for i in batch_idx], dtype=torch.float)
        query_idx = [torch.tensor(features[i].query_idx, dtype=torch.long) for i in batch_idx]
        doc_idx = [torch.tensor(features[i].doc_idx, dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i].query_len for i in batch_idx], dtype=torch.long)
        doc_len = torch.tensor([features[i].doc_len for i in batch_idx], dtype=torch.long)

        query_idx = nn.utils.rnn.pad_sequence(query_idx, batch_first=True)
        doc_idx = nn.utils.rnn.pad_sequence(doc_idx, batch_first=True)

        batch = (query_id, doc_id, query, doc, raw_score, query_idx, doc_idx, query_len, doc_len)
        batches.append(batch)
    return batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', help='embedding path (glove embedding)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--test_file', help='test file path')
    parser.add_argument('--out_path', help='out path to save trec file')
    parser.add_argument('--ensemble', default=False, help='ensemble or not')
    parser.add_argument('--pretrained_model', help='check point to load')
    parser.add_argument('--vocab_size', default=400002, type=int, help='vocab size with padding and unk words')
    parser.add_argument('--embedding_dim', default=300, type=int, help='embedding dim')
    parser.add_argument('--kernel_size', default=21, type=int, help='kernel size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--cnn_kernel', default=128, type=int, help='cnn kernel size')
    parser.add_argument('--max_query_len', default=20, type=int, help='max query length')
    parser.add_argument('--max_doc_len', default=128, type=int, help='max doc length')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    idx2word, word2idx, word2vec = load_glove(args.embedding_path)
    embedding_matrix = create_embeddings(idx2word, word2vec)

    model = Ranker(embedding_matrix, args)
    state_dict=torch.load(args.pretrained_model)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)

    # test data
    test_features = read_data_to_features(args.test_file, word2idx, args)
    test_data = devDataLoader(test_features, args.batch_size)

    # test
    rst_dict = {}
    for s, batch in enumerate(test_data):
        query_id = batch[0]
        doc_id = batch[1]
        query = batch[2]
        doc = batch[3]
        batch = tuple(t.to(device) for t in batch[4:])
        (raw_score, query_idx, doc_idx, query_len, doc_len) = batch

        with torch.no_grad():
            doc_scores, doc_features = model(query_idx, doc_idx, None, query_len, doc_len, None, raw_score, False)
        d_scores = doc_scores.detach().cpu().tolist()
        d_features = doc_features.detach().cpu().tolist()
        raw_score = raw_score.detach().cpu().tolist()
        for (q_id, d_id, q, d, r_s, d_s, d_f) in zip(query_id, doc_id, query, doc, raw_score, d_scores, d_features):
            if q_id in rst_dict:
                rst_dict[q_id].append((d_s, d_id, q, d))
            else:
                rst_dict[q_id] = [(d_s, d_id, q, d)]

    with open(args.out_path, 'w') as writer:
        tmp = {"query_id": "", "records": []}
        for q_id, records in rst_dict.items():
            tmp["query_id"] = q_id
            tmp["query"] = rst_dict[q_id][0][2]
            tmp["records"] = []
            max_pool = []
            res = sorted(records, key=lambda x: x[0], reverse=True)
            for rank, value in enumerate(res):
                if value[1] not in max_pool:
                    max_pool.append(value[1])
                    tmp["records"].append({"paper_id":value[1], "score":value[0], "paragraph":value[3]})
            writer.write(json.dumps(tmp) + '\n')

if __name__ == "__main__":
    main()
