import numpy as np
import torch
from torch import nn

def read_train_to_features(args, word2idx):
    with open(args.train, 'r') as reader:
        features = []
        for line in reader:
            s = line.strip('\n').split('\t')

            query_toks = s[0].split()
            pos_toks = s[1].split()
            neg_toks = s[2].split()

            query_toks = stopword_removal(query_toks)
            pos_toks = stopword_removal(pos_toks)
            neg_toks = stopword_removal(neg_toks)

            query_toks = stemming(query_toks)
            pos_toks = stemming(pos_toks)
            neg_toks = stemming(neg_toks)

            query_toks = query_toks[:args.max_query_len]
            pos_toks = pos_toks[:args.max_seq_len]
            neg_toks = neg_toks[:args.max_seq_len]

            query_len = len(query_toks)
            pos_len = len(pos_toks)
            neg_len = len(neg_toks)

            while len(query_toks) < args.max_query_len:
                query_toks.append('<PAD>')
            while len(pos_toks) < args.max_seq_len:
                pos_toks.append('<PAD>')
            while len(neg_toks) < args.max_seq_len:
                neg_toks.append('<PAD>')

            query_idx = tok2idx(query_toks, word2idx)
            pos_idx = tok2idx(pos_toks, word2idx)
            neg_idx = tok2idx(neg_toks, word2idx)

            features.append({
                'query_idx': query_idx,
                'pos_idx': pos_idx,
                'neg_idx': neg_idx,
                'query_len': query_len,
                'pos_len': pos_len,
                'neg_len': neg_len})
        return features

def read_dev_to_features(args, word2idx):
    with open(args.dev, 'r') as reader:
        features = []
        for line in reader:
            s = line.strip('\n').split('\t')

            query_toks = s[0].split()
            doc_toks = s[1].split()
            label = int(s[2])
            query_id = s[3]
            doc_id = s[4]
            retrieval_score = float(s[5])

            query_toks = stopword_removal(query_toks)
            doc_toks = stopword_removal(doc_toks)

            query_toks = stemming(query_toks)
            doc_toks = stemming(doc_toks)

            query_toks = query_toks[:args.max_query_len]
            doc_toks = doc_toks[:args.max_seq_len]

            query_len = len(query_toks)
            doc_len = len(doc_toks)

            while len(query_toks) < args.max_query_len:
                query_toks.append('<PAD>')
            while len(doc_toks) < args.max_seq_len:
                doc_toks.append('<PAD>')

            query_idx = tok2idx(query_toks, word2idx)
            doc_idx = tok2idx(doc_toks, word2idx)

            features.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,
                'retrieval_score': retrieval_score,
                'query_idx': query_idx,
                'doc_idx': doc_idx,
                'query_len': query_len,
                'doc_len': doc_len})
        return features

def train_dataloader(args, tokenizer, shuffle=True):
    features = read_train_to_features(args, tokenizer)
    n_samples = len(features)
    idx = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idx)

    for start_idx in range(0, n_samples, args.batch_size):
        batch_idx = idx[start_idx:start_idx+args.batch_size]

        query_idx = [torch.tensor(features[i]['query_idx'], dtype=torch.long) for i in batch_idx]
        pos_idx = [torch.tensor(features[i]['pos_idx'], dtype=torch.long) for i in batch_idx]
        neg_idx = [torch.tensor(features[i]['neg_idx'], dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i]['query_len'] for i in batch_idx], dtype=torch.long)
        pos_len = torch.tensor([features[i]['pos_len'] for i in batch_idx], dtype=torch.long)
        neg_len = torch.tensor([features[i]['neg_len'] for i in batch_idx], dtype=torch.long)

        query_idx = nn.utils.rnn.pad_sequence(query_idx, batch_first=True)
        pos_idx = nn.utils.rnn.pad_sequence(pos_idx, batch_first=True)
        neg_idx = nn.utils.rnn.pad_sequence(neg_idx, batch_first=True)

        batch = (query_idx, pos_idx, neg_idx, query_len, pos_len, neg_len)
        yield batch
    return

def dev_dataloader(args, tokenizer):
    features = read_dev_to_features(args, tokenizer)
    n_samples = len(features)
    idx = np.arange(n_samples)
    batches = []
    for start_idx in range(0, n_samples, args.batch_size):
        batch_idx = idx[start_idx:start_idx+args.batch_size]

        query_id = [features[i]['query_id'] for i in batch_idx]
        doc_id = [features[i]['doc_id'] for i in batch_idx]
        label = [features[i]['label'] for i in batch_idx]
        retrieval_score = torch.tensor([features[i]['retireval_score'] for i in batch_idx], dtype=torch.float)
        query_idx = [torch.tensor(features[i]['query_idx'], dtype=torch.long) for i in batch_idx]
        doc_idx = [torch.tensor(features[i]['doc_idx'], dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i]['query_len'] for i in batch_idx], dtype=torch.long)
        doc_len = torch.tensor([features[i]['doc_len'] for i in batch_idx], dtype=torch.long)

        query_idx = nn.utils.rnn.pad_sequence(query_idx, batch_first=True)
        doc_idx = nn.utils.rnn.pad_sequence(doc_idx, batch_first=True)

        batch = (query_id, doc_id, label, retrieval_score, query_idx, doc_idx, query_len, doc_len)
        batches.append(batch)
    return batches
