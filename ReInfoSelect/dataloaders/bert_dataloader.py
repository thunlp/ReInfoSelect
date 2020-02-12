import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

class trainFeatures(object):
    def __init__(self, query_idx, doc_idx, query_len, doc_len, p_input_ids, p_input_mask, p_segment_ids, n_input_ids, n_input_mask, n_segment_ids):
        self.query_idx = query_idx
        self.doc_idx = doc_idx
        self.query_len = query_len
        self.doc_len = doc_len
        self.p_input_ids = p_input_ids
        self.p_input_mask = p_input_mask
        self.p_segment_ids = p_segment_ids
        self.n_input_ids = n_input_ids
        self.n_input_mask = n_input_mask
        self.n_segment_ids = n_segment_ids

class devFeatures(object):
    def __init__(self, query_id, doc_id, qd_score, raw_score, d_input_ids, d_input_mask, d_segment_ids):
        self.query_id = query_id
        self.doc_id = doc_id
        self.qd_score = qd_score
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

def tok2idx(toks, word2idx):
    input_ids = []
    for tok in toks:
        if tok in word2idx:
            input_ids.append(word2idx[tok])
        else:
            input_ids.append(word2idx['<UNK>'])
    return input_ids

def read_train_to_features(args, word2idx, tokenizer):
    with open(args.train, 'r') as reader:
        features = []
        for line in reader:
            s = line.strip('\n').split('\t')

            query_toks = s[0].split()
            pos_toks = s[1].split()
            neg_toks = s[2].split()

            query_toks = query_toks[:20]
            pos_toks = pos_toks[:args.max_seq_len]
            neg_toks = neg_toks[:args.max_seq_len]

            query_len = len(query_toks)
            pos_len = len(pos_toks)
            neg_len = len(neg_toks)

            while len(query_toks) < 20:
                query_toks.append('<PAD>')
            while len(pos_toks) < args.max_seq_len:
                pos_toks.append('<PAD>')
            while len(neg_toks) < args.max_seq_len:
                neg_toks.append('<PAD>')

            query_idx = tok2idx(query_toks, word2idx)
            pos_idx = tok2idx(pos_toks, word2idx)
            neg_idx = tok2idx(neg_toks, word2idx)

            s[0] = ' '.join(query_toks)
            s[1] = ' '.join(pos_toks)
            s[2] = ' '.join(neg_toks)
            q_tokens = tokenizer.tokenize(s[0])
            if len(q_tokens) > 64:
                q_tokens = q_tokens[:64]
            max_doc_length = 384 - len(q_tokens) - 3
            p_tokens = tokenizer.tokenize(s[1])
            if len(p_tokens) > max_doc_length:
                p_tokens = p_tokens[:max_doc_length]
            n_tokens = tokenizer.tokenize(s[2])
            if len(n_tokens) > max_doc_length:
                n_tokens = n_tokens[:max_doc_length]

            p_input_ids, p_input_mask, p_segment_ids = pack_bert_seq(q_tokens, p_tokens, tokenizer, 384)
            n_input_ids, n_input_mask, n_segment_ids = pack_bert_seq(q_tokens, n_tokens, tokenizer, 384)

            features.append(trainFeatures(
                query_idx = query_idx,
                doc_idx = pos_idx,
                query_len = query_len,
                doc_len = pos_len,
                p_input_ids = p_input_ids,
                p_input_mask = p_input_mask,
                p_segment_ids = p_segment_ids,
                n_input_ids = n_input_ids,
                n_input_mask = n_input_mask,
                n_segment_ids = n_segment_ids))
        return features

def read_dev_to_features(args, tokenizer):
    with open(args.dev, 'r') as reader:
        features = []
        for line in reader:
            s = line.strip('\n').split('\t')

            query_toks = s[0].split()
            doc_toks = s[1].split()
            qd_score = int(s[2])
            query_id = s[3]
            doc_id = s[4]
            raw_score = float(s[5])

            query_toks = query_toks[:20]
            doc_toks = doc_toks[:args.max_seq_len]

            query_len = len(query_toks)
            doc_len = len(doc_toks)

            while len(query_toks) < 20:
                query_toks.append('<PAD>')
            while len(doc_toks) < args.max_seq_len:
                doc_toks.append('<PAD>')

            s[0] = ' '.join(query_toks)
            s[1] = ' '.join(doc_toks)

            q_tokens = tokenizer.tokenize(s[0])
            if len(q_tokens) > 64:
                q_tokens = q_tokens[:64]
            max_doc_length = 384 - len(q_tokens) - 3
            d_tokens = tokenizer.tokenize(s[1])
            if len(d_tokens) > max_doc_length:
                d_tokens = d_tokens[:max_doc_length]

            d_input_ids, d_input_mask, d_segment_ids = pack_bert_seq(q_tokens, d_tokens, tokenizer, 384)

            features.append(devFeatures(
                query_id = query_id,
                doc_id = doc_id,
                qd_score = qd_score,
                raw_score = raw_score,
                d_input_ids = d_input_ids,
                d_input_mask = d_input_mask,
                d_segment_ids = d_segment_ids))
        return features

def bert_train_dataloader(args, word2idx, tokenizer, shuffle=True):
    features = read_train_to_features(args, word2idx, tokenizer)
    n_samples = len(features)
    idx = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idx)
    batches = []
    for start_idx in range(0, n_samples, args.batch_size):
        batch_idx = idx[start_idx:start_idx+args.batch_size]

        query_idx = [torch.tensor(features[i].query_idx, dtype=torch.long) for i in batch_idx]
        doc_idx = [torch.tensor(features[i].doc_idx, dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i].query_len for i in batch_idx], dtype=torch.long)
        doc_len = torch.tensor([features[i].doc_len for i in batch_idx], dtype=torch.long)
        p_input_ids = torch.tensor([features[i].p_input_ids for i in batch_idx], dtype=torch.long)
        p_input_mask = torch.tensor([features[i].p_input_mask for i in batch_idx], dtype=torch.long)
        p_segment_ids = torch.tensor([features[i].p_segment_ids for i in batch_idx], dtype=torch.long)
        n_input_ids = torch.tensor([features[i].n_input_ids for i in batch_idx], dtype=torch.long)
        n_input_mask = torch.tensor([features[i].n_input_mask for i in batch_idx], dtype=torch.long)
        n_segment_ids = torch.tensor([features[i].n_segment_ids for i in batch_idx], dtype=torch.long)

        query_idx = nn.utils.rnn.pad_sequence(query_idx, batch_first=True)
        doc_idx = nn.utils.rnn.pad_sequence(doc_idx, batch_first=True)

        batch = (query_idx, doc_idx, query_len, doc_len, p_input_ids, p_input_mask, p_segment_ids, n_input_ids, n_input_mask, n_segment_ids)
        batches.append(batch)
    return batches

def bert_dev_dataloader(args, tokenizer):
    features = read_dev_to_features(args, tokenizer)
    n_samples = len(features)
    idx = np.arange(n_samples)
    batches = []
    for start_idx in range(0, n_samples, args.batch_size):
        batch_idx = idx[start_idx:start_idx+args.batch_size]

        query_id = [features[i].query_id for i in batch_idx]
        doc_id = [features[i].doc_id for i in batch_idx]
        qd_score = [features[i].qd_score for i in batch_idx]
        raw_score = torch.tensor([features[i].raw_score for i in batch_idx], dtype=torch.float)
        d_input_ids = torch.tensor([features[i].d_input_ids for i in batch_idx], dtype=torch.long)
        d_input_mask = torch.tensor([features[i].d_input_mask for i in batch_idx], dtype=torch.long)
        d_segment_ids = torch.tensor([features[i].d_segment_ids for i in batch_idx], dtype=torch.long)

        batch = (query_id, doc_id, qd_score, raw_score, d_input_ids, d_input_mask, d_segment_ids)
        batches.append(batch)
    return batches
