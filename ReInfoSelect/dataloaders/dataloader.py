import numpy as np
import torch
from torch import nn

def read_train_to_features(args, tokenizer):
    with open(args.train, 'r') as reader:
        features = []
        for line in reader:
            if len(features) == args.max_input:
                break
            s = line.strip('\n').split('\t')

            query_toks = tokenizer.tokenize(s[0])[:args.max_query_len]
            pos_toks = tokenizer.tokenize(s[1])[:args.max_seq_len]
            neg_toks = tokenizer.tokenize(s[2])[:args.max_seq_len]

            query_len = len(query_toks)
            pos_len = len(pos_toks)
            neg_len = len(neg_toks)

            while len(query_toks) < args.max_query_len:
                query_toks.append(tokenizer.pad)
            while len(pos_toks) < args.max_seq_len:
                pos_toks.append(tokenizer.pad)
            while len(neg_toks) < args.max_seq_len:
                neg_toks.append(tokenizer.pad)

            query_idx = tokenizer.convert_tokens_to_ids(query_toks)
            pos_idx = tokenizer.convert_tokens_to_ids(pos_toks)
            neg_idx = tokenizer.convert_tokens_to_ids(neg_toks)

            features.append({
                'query_idx': query_idx,
                'pos_idx': pos_idx,
                'neg_idx': neg_idx,
                'query_len': query_len,
                'pos_len': pos_len,
                'neg_len': neg_len})
        return features

def read_dev_to_features(args, tokenizer):
    with open(args.dev, 'r') as reader:
        features = []
        for line in reader:
            if len(features) == args.max_input:
                break
            s = line.strip('\n').split('\t')

            label = int(s[2])
            query_id = s[3]
            doc_id = s[4]
            retrieval_score = float(s[5])

            query_toks = tokenizer.tokenize(s[0])[:args.max_query_len]
            doc_toks = tokenizer.tokenize(s[1])[:args.max_seq_len]

            query_len = len(query_toks)
            doc_len = len(doc_toks)

            while len(query_toks) < args.max_query_len:
                query_toks.append(tokenizer.pad)
            while len(doc_toks) < args.max_seq_len:
                doc_toks.append(tokenizer.pad)

            query_idx = tokenizer.convert_tokens_to_ids(query_toks)
            doc_idx = tokenizer.convert_tokens_to_ids(doc_toks)

            features.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,
                'query': s[0],
                'doc': s[1],
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
        query = [features[i]['query'] for i in batch_idx]
        doc = [features[i]['doc'] for i in batch_idx]
        retrieval_score = torch.tensor([features[i]['retrieval_score'] for i in batch_idx], dtype=torch.float)
        query_idx = [torch.tensor(features[i]['query_idx'], dtype=torch.long) for i in batch_idx]
        doc_idx = [torch.tensor(features[i]['doc_idx'], dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i]['query_len'] for i in batch_idx], dtype=torch.long)
        doc_len = torch.tensor([features[i]['doc_len'] for i in batch_idx], dtype=torch.long)

        query_idx = nn.utils.rnn.pad_sequence(query_idx, batch_first=True)
        doc_idx = nn.utils.rnn.pad_sequence(doc_idx, batch_first=True)

        batch = (query_id, doc_id, label, query, doc, retrieval_score, query_idx, doc_idx, query_len, doc_len)
        batches.append(batch)
    return batches
