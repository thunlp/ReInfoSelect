import numpy as np
import torch
from torch import nn

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

def read_train_to_features(args, tokenizer, bert_tokenizer):
    with open(args.train, 'r') as reader:
        features = []
        for line in reader:
            if len(features) >= args.max_input:
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

            s[0] = ' '.join(query_toks)
            s[1] = ' '.join(pos_toks)
            s[2] = ' '.join(neg_toks)
            q_tokens = bert_tokenizer.tokenize(s[0])
            if len(q_tokens) > 64:
                q_tokens = q_tokens[:64]
            max_doc_length = 384 - len(q_tokens) - 3
            p_tokens = bert_tokenizer.tokenize(s[1])
            if len(p_tokens) > max_doc_length:
                p_tokens = p_tokens[:max_doc_length]
            n_tokens = bert_tokenizer.tokenize(s[2])
            if len(n_tokens) > max_doc_length:
                n_tokens = n_tokens[:max_doc_length]

            p_input_ids, p_input_mask, p_segment_ids = pack_bert_seq(q_tokens, p_tokens, bert_tokenizer, 384)
            n_input_ids, n_input_mask, n_segment_ids = pack_bert_seq(q_tokens, n_tokens, bert_tokenizer, 384)

            features.append({
                'query_idx': query_idx,
                'doc_idx': pos_idx,
                'query_len': query_len,
                'doc_len': pos_len,
                'p_input_ids': p_input_ids,
                'p_input_mask': p_input_mask,
                'p_segment_ids': p_segment_ids,
                'n_input_ids': n_input_ids,
                'n_input_mask': n_input_mask,
                'n_segment_ids': n_segment_ids})
        return features

def read_dev_to_features(args, tokenizer):
    with open(args.dev, 'r') as reader:
        features = []
        for line in reader:
            if len(features) >= args.max_input:
                break
            s = line.strip('\n').split('\t')

            label = int(s[2])
            query_id = s[3]
            doc_id = s[4]
            retrieval_score = float(s[5])

            q_tokens = tokenizer.tokenize(s[0])
            if len(q_tokens) > 64:
                q_tokens = q_tokens[:64]
            max_doc_length = 384 - len(q_tokens) - 3
            d_tokens = tokenizer.tokenize(s[1])
            if len(d_tokens) > max_doc_length:
                d_tokens = d_tokens[:max_doc_length]

            d_input_ids, d_input_mask, d_segment_ids = pack_bert_seq(q_tokens, d_tokens, tokenizer, 384)

            features.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,
                'query': s[0],
                'doc': s[1],
                'retrieval_score': retrieval_score,
                'd_input_ids': d_input_ids,
                'd_input_mask': d_input_mask,
                'd_segment_ids': d_segment_ids})
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

        query_idx = [torch.tensor(features[i]['query_idx'], dtype=torch.long) for i in batch_idx]
        doc_idx = [torch.tensor(features[i]['doc_idx'], dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i]['query_len'] for i in batch_idx], dtype=torch.long)
        doc_len = torch.tensor([features[i]['doc_len'] for i in batch_idx], dtype=torch.long)
        p_input_ids = torch.tensor([features[i]['p_input_ids'] for i in batch_idx], dtype=torch.long)
        p_input_mask = torch.tensor([features[i]['p_input_mask'] for i in batch_idx], dtype=torch.long)
        p_segment_ids = torch.tensor([features[i]['p_segment_ids'] for i in batch_idx], dtype=torch.long)
        n_input_ids = torch.tensor([features[i]['n_input_ids'] for i in batch_idx], dtype=torch.long)
        n_input_mask = torch.tensor([features[i]['n_input_mask'] for i in batch_idx], dtype=torch.long)
        n_segment_ids = torch.tensor([features[i]['n_segment_ids'] for i in batch_idx], dtype=torch.long)

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

        query_id = [features[i]['query_id'] for i in batch_idx]
        doc_id = [features[i]['doc_id'] for i in batch_idx]
        label = [features[i]['label'] for i in batch_idx]
        query = [features[i]['query'] for i in batch_idx]
        doc = [features[i]['doc'] for i in batch_idx]
        retrieval_score = torch.tensor([features[i]['retrieval_score'] for i in batch_idx], dtype=torch.float)
        d_input_ids = torch.tensor([features[i]['d_input_ids'] for i in batch_idx], dtype=torch.long)
        d_input_mask = torch.tensor([features[i]['d_input_mask'] for i in batch_idx], dtype=torch.long)
        d_segment_ids = torch.tensor([features[i]['d_segment_ids'] for i in batch_idx], dtype=torch.long)

        batch = (query_id, doc_id, label, query, doc, retrieval_score, d_input_ids, d_input_mask, d_segment_ids)
        batches.append(batch)
    return batches
