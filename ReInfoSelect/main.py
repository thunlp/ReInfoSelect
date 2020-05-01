import argparse
import json

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Categorical

from cknrm_tokenizer import *
from policies import *
from dataloaders import *
from models import *
from metrics import *

from transformers import BertTokenizer, BertConfig, AdamW

def dev(args, model, dev_data, device):
    features = []
    rst_dict = {}
    for s, batch in enumerate(dev_data):
        query_id = batch[0]
        doc_id = batch[1]
        label = batch[2]
        query = batch[3]
        doc = batch[4]
        batch = tuple(t.to(device) for t in batch[5:])
        if args.model.lower() == 'bert':
            (retrieval_score, d_input_ids, d_input_mask, d_segment_ids) = batch
            with torch.no_grad():
                doc_scores, doc_features = model(d_input_ids, d_input_mask, d_segment_ids, retrieval_score)
        elif args.model.lower() == 'cknrm':
            (retrieval_score, query_idx, doc_idx, query_len, doc_len) = batch
            with torch.no_grad():
                doc_scores, doc_features = model(query_idx, doc_idx, query_len, doc_len, retrieval_score)
        else:
            raise ('model must be bert or cknrm!')
        d_scores = doc_scores.detach().cpu().tolist()
        d_features = doc_features.detach().cpu().tolist()

        for (q_id, d_id, l_s, q, d, d_s, d_f) in zip(query_id, doc_id, label, query, doc, d_scores, d_features):
            feature = []
            feature.append(str(l_s))
            feature.append('id:' + q_id)
            for i, fi in enumerate(d_f):
                feature.append(str(i+1) + ':' + str(fi))
            features.append(' '.join(feature))
            if q_id in rst_dict:
                rst_dict[q_id].append((l_s, d_s, d_id, q, d))
            else:
                rst_dict[q_id] = [(l_s, d_s, d_id, q, d)]

    with open(args.res_trec, 'w') as writer:
        for q_id, scores in rst_dict.items():
            ps = {}
            res = sorted(scores, key=lambda x: x[1], reverse=True)
            for rank, value in enumerate(res):
                if value[2] not in ps:
                    ps[value[2]] = 1
                    writer.write(q_id+' '+'Q0'+' '+str(value[2])+' '+str(rank+1)+' '+str(value[1])+' bert\n')

    with open(args.res_json, 'w') as writer:
        tmp = {"query_id": "", "records": []}
        for q_id, scores in rst_dict.items():
            tmp["query_id"] = q_id
            tmp["records"] = []
            ps = {}
            res = sorted(scores, key=lambda x: x[1], reverse=True)
            tmp["query"] = res[0][3]
            for rank, value in enumerate(res):
                if value[2] not in ps:
                    ps[value[2]] = 1
                    tmp["records"].append({"paper_id":value[2], "score":value[1], "paragraph":value[4]})
            writer.write(json.dumps(tmp) + '\n')

    ndcg = cal_ndcg(args.qrels, args.res_trec, args.depth)
    return ndcg, features

def train(args, policy, p_optim, model, m_optim, crit, tokenizer, bert_tokenizer, dev_data, device):
    best_ndcg = 0.0
    for ep in range(args.epoch):
        # train data
        if args.model.lower() == 'bert':
            train_data = bert_train_dataloader(args, tokenizer, bert_tokenizer)
        elif args.model.lower() == 'cknrm':
            train_data = train_dataloader(args, tokenizer)
        else:
            raise ('model must be bert or cknrm!')
        ndcg, features = dev(args, model, dev_data, device)
        print('init_ndcg: ' + str(ndcg))
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            with open(args.res_feature, 'w') as writer:
                for feature in features:
                    writer.write(feature+'\n')
            print('save best model on target dataset...')
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), args.save_best)
            else:
                torch.save(model.state_dict(), args.save_best)
        last_ndcg = ndcg

        log_prob_ps = []
        log_prob_ns = []
        log_probs = []
        rewards = []
        for step, batch in enumerate(train_data):
            # select action
            batch = tuple(t.to(device) for t in batch)
            if args.model.lower() == 'bert':
                (query_idx, doc_idx, query_len, doc_len, p_input_ids, p_input_mask, p_segment_ids, n_input_ids, n_input_mask, n_segment_ids) = batch
            elif args.model.lower() == 'cknrm':
                (query_idx, doc_idx, neg_idx, query_len, doc_len, neg_len) = batch
            else:
                raise ('model must be bert or cknrm!')

            probs = policy(query_idx, doc_idx, query_len, doc_len)
            dist  = Categorical(probs)
            action = dist.sample()
            if action.sum().item() < 1 and (step % args.T != 0 or len(rewards) == 0):
                continue

            mask = action.ge(0.5)
            weights = Variable(action, requires_grad=False).float().cuda()
            log_prob_p = dist.log_prob(action)
            log_prob_n = dist.log_prob(1-action)
            log_prob_ps.append(torch.masked_select(log_prob_p, mask))
            log_prob_ns.append(torch.masked_select(log_prob_n, mask))

            if args.model.lower() == 'bert':
                p_scores, _ = model(p_input_ids, p_segment_ids, p_input_mask)
                n_scores, _ = model(n_input_ids, n_segment_ids, n_input_mask)
            elif args.model.lower() == 'cknrm':
                p_scores, _ = model(query_idx, doc_idx, query_len, doc_len)
                n_scores, _ = model(query_idx, neg_idx, query_len, neg_len)

            label = torch.ones(p_scores.size()).to(device)
            batch_loss = crit(p_scores, n_scores, Variable(label, requires_grad=False))
            batch_loss = batch_loss.mul(weights).mean()
            batch_loss.backward()
            m_optim.step()
            m_optim.zero_grad()

            ndcg, features = dev(args, model, dev_data, device)
            print('epoch: ' + str(ep+1) + ', step: ' + str(step+1) + ', ndcg: ' + str(last_ndcg) + ', best_ndcg: ' + str(best_ndcg))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                with open(args.res_feature, 'w') as writer:
                    for feature in features:
                        writer.write(feature+'\n')
                print('save best model on target dataset...')
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), args.save_best)
                else:
                    torch.save(model.state_dict(), args.save_best)
            reward = ndcg - last_ndcg
            last_ndcg = ndcg
            rewards.append(reward)

            if len(rewards) > 0 and step % args.T == 0:
                R = 0.0
                policy_loss = []
                returns = []
                for ri in reversed(range(len(rewards))):
                    R = rewards[ri] + args.gamma * R
                    if R > 0:
                        log_probs.insert(0, log_prob_ps[ri])
                        returns.insert(0, R)
                    else:
                        log_probs.insert(0, log_prob_ns[ri])
                        returns.insert(0, -R)
                for lp, r in zip(log_probs, returns):
                    policy_loss.append((-lp * r).sum().unsqueeze(-1))
                loss = torch.cat(policy_loss).sum()
                loss.backward()
                p_optim.step()
                p_optim.zero_grad()

                log_prob_ps = []
                log_prob_ns = []
                log_probs = []
                rewards = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-train', type=str, default='../data/triples.train.small.tsv')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save_best', type=str, default='../checkpoints/reinfoselect_bert.bin')
    parser.add_argument('-dev', type=str, default='../data/dev_toy.tsv')
    parser.add_argument('-qrels', type=str, default='../data/qrels_toy')
    parser.add_argument('-embed', type=str, default='../data/glove.6B.300d.txt')
    parser.add_argument('-pretrain', type=str, default='bert-base-uncased')
    parser.add_argument('-vocab_size', type=int, default=400002)
    parser.add_argument('-embed_dim', type=int, default=300)
    parser.add_argument('-res_trec', type=str, default='../results/bert.trec')
    parser.add_argument('-res_json', type=str, default='../results/cknrm.json')
    parser.add_argument('-res_feature', type=str, default='../features/bert_features')
    parser.add_argument('-depth', type=int, default=20)
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-T', type=int, default=4)
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_seq_len', type=int, default=150)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=4)
    args = parser.parse_args()

    # init embedding
    embedding_init = embeddingloader(args)
    tokenizer = Tokenizer(args.embed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init policy
    policy = Policy(args, embedding_init)
    policy.to(device)
    p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=1e-4)

    if args.model.lower() == 'bert':
        bert_tokenizer = BertTokenizer.from_pretrained(args.pretrain)
        config = BertConfig.from_pretrained(args.pretrain)
        model = BertForRanking.from_pretrained(args.pretrain, config=config)

        m_optim = AdamW(model.parameters(), lr=5e-5)
        dev_data = bert_dev_dataloader(args, bert_tokenizer)
    elif args.model.lower() == 'cknrm':
        bert_tokenizer = None
        model = cknrm(args, embedding_init)

        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        dev_data = dev_dataloader(args, tokenizer)
    else:
        raise ('model must be bert or cknrm!')
    model.to(device)

    # loss function
    crit = nn.MarginRankingLoss(margin=1, reduction='mean')
    crit.to(device)

    if torch.cuda.device_count() > 1:
        policy = nn.DataParallel(policy)
        model = nn.DataParallel(model)
        crit = nn.DataParallel(crit)

    if args.mode == 'train':
        train(args, policy, p_optim, model, m_optim, crit, tokenizer, bert_tokenizer, dev_data, device)
    elif args.mode == 'infer':
        assert args.checkpoint is not None
        state_dict=torch.load(args.checkpoint)
        model.load_state_dict(state_dict)
        ndcg, features = dev(args, model, dev_data, device)
        with open(args.res_feature, 'w') as writer:
            for feature in features:
                writer.write(feature+'\n')
    else:
        raise ('mode must be train or infer!')

if __name__ == "__main__":
    main()
