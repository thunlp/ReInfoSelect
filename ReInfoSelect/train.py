import yaml
import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from dataloaders import *
from metrics import *
from policy import *
from ranker import *

def dev(cfg, model, dev_data, device):
    rst_dict = {}
    for s, batch in enumerate(dev_data):
        query_id = batch[0]
        doc_id = batch[1]
        qd_score = batch[2]
        batch = tuple(t.to(device) for t in batch[3:])
        (score_feature, query_idx, doc_idx, query_len, doc_len) = batch

        with torch.no_grad():
            if cfg["score_feature"]:
                doc_scores = model(query_idx, doc_idx, query_len, doc_len, score_feature)
            else:
                doc_scores = model(query_idx, doc_idx, query_len, doc_len, None)
        d_scores = doc_scores.detach().cpu().tolist()

        for (q_id, d_id, qd_s, d_s) in zip(query_id, doc_id, qd_score, d_scores):
            if q_id in rst_dict:
                rst_dict[q_id].append((qd_s, d_s, d_id))
            else:
                rst_dict[q_id] = [(qd_s, d_s, d_id)]

    with open(cfg["out_trec"], 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores, key=lambda x: x[1], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id+' '+'Q0'+' '+str(value[2])+' '+str(rank+1)+' '+str(value[1])+' '+cfg["model"]+'\n')

    m_ndcg = ndcg(cfg["label"], cfg["out_trec"], cfg["depth"])
    m_err = err(cfg["label"], cfg["out_trec"], cfg["depth"])
    measure = [m_ndcg, m_err]
    return measure

def train(cfg, policy, p_optim, model, m_optim, crit, word2vec, tokenizer, dev_data, device):
    best_ndcg = 0.0
    for i_episode in range(cfg["epoch"]):
        # train data
        if cfg["model"] != "bert":
            train_data = train_dataloader(cfg, word2vec)
        else:
            train_data = bert_train_dataloader(cfg, word2vec, tokenizer)
        if cfg["policy"]:
            ndcg, err = dev(cfg, model, dev_data, device)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
            last_ndcg = ndcg

            log_prob_ps = []
            log_prob_ns = []
            log_probs = []
            rewards = []
        for step, batch in enumerate(train_data):
            # select action
            batch = tuple(t.to(device) for t in batch)
            (query_idx, pos_idx, neg_idx, query_len, pos_len, neg_len) = batch

            if cfg["policy"]:
                probs = policy(query_idx, pos_idx, query_len, pos_len)
                dist  = Categorical(probs)
                action = dist.sample()
                if action.sum().item() < 1:
                    continue
            else:
                action = torch.tensor([1]*cfg["batch_size"], dtype=torch.long).to(device)

            filt = torch.nonzero(action.squeeze(-1)).squeeze(-1)
            probs = torch.index_select(probs, 0, filt)
            log_prob_p = torch.log(probs[:, 1])
            log_prob_n = torch.log(probs[:, 0])

            query_idx_f = torch.index_select(query_idx, 0, filt)
            pos_idx_f = torch.index_select(pos_idx, 0, filt)
            neg_idx_f = torch.index_select(neg_idx, 0, filt)
            query_len_f = torch.index_select(query_len, 0, filt)
            pos_len_f = torch.index_select(pos_len, 0, filt)
            neg_len_f = torch.index_select(neg_len, 0, filt)

            p_scores = model(query_idx_f, pos_idx_f, query_len_f, pos_len_f)
            n_scores = model(query_idx_f, neg_idx_f, query_len_f, neg_len_f)
            label = torch.ones(p_scores.size()).to(device)
            batch_loss = crit(p_scores, n_scores, Variable(label, requires_grad=False))
            batch_loss.backward()
            m_optim.step()
            m_optim.zero_grad()

            if cfg["policy"]:
                log_prob_ps.append(log_prob_p)
                log_prob_ns.append(log_prob_n)

                ndcg, err = dev(cfg, model, dev_data, device)
                if ndcg > best_ndcg:
                     best_ndcg = ndcg
                reward = ndcg - last_ndcg
                last_ndcg = ndcg
                rewards.append(reward)

            if cfg["policy"] and len(rewards) > 0 and step % cfg["k"] == 0:
                print('update policy...')
                R = 0.0
                policy_loss = []
                returns = []
                for ri in reversed(range(len(rewards))):
                    R = rewards[ri] + gamma_raw * R
                    if R > 0:
                        log_probs.insert(0, log_prob_ps[ri])
                        returns.insert(0, R)
                    else:
                        log_probs.insert(0, log_prob_ns[ri])
                        returns.insert(0, -R)
                for lp, r in zip(log_probs, returns):
                    policy_loss.append((-lp * r).sum().unsqueeze(-1))
                p_optim.zero_grad()
                loss = torch.cat(policy_loss).sum()
                loss.backward()
                p_optim.step()

                log_prob_ps = []
                log_prob_ns = []
                log_probs = []
                rewards = []

            if step % cfg["check_step"] == 0:
                ndcg, err = dev(cfg, model, dev_data, device)
                if ndcg > best_ndcg:
                     best_ndcg = ndcg
                print("current ndcg: " + str(ndcg) + ", best ndcg: " + str(best_ndcg))

        if cfg["multi_gpu"]:
            torch.save(policy.module.state_dict(), save_policy)
            torch.save(model.module.state_dict(), save_model)
        else:
            torch.save(policy.state_dict(), save_policy)
            torch.save(model.state_dict(), save_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", default="./configs/config_example.yaml", required=True)
    args = parser.parse_args()

    cfg = {}
    with open(args.config, 'r') as r:
        cfg.update(yaml.load(r))

    # init embedding
    if cfg["model"] != "bert":
        word2vec, embedding_init = embloader(cfg)
        tokenizer = None
    else:
        if cfg["policy"]:
            word2vec, embedding_init = embloader(cfg)
        else:
            word2vec = None
            embedding_init = None
        tokenizer = bert_embloader(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init policy
    if cfg["policy"]:
        if cfg["policy"] == "q":
            policy = q_policy(cfg, embedding_init)
        elif cfg["policy"] == "qd":
            policy = qd_policy(cfg, embedding_init)
        elif cfg["policy"] == "all":
            policy = all_policy(cfg, embedding_init)
        else:
            print("Policy Error!")
            exit()
        policy.to(device)
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=0.001)
    else:
        policy = None
        p_optim = None

    # load trained policy
    if cfg["classify"]:
        state_dict=torch.load(cfg["load_policy"])
        policy.load_state_dict(state_dict)

    # init model
    if cfg["model"] == "knrm":
        model = knrm(cfg, embedding_init)
    elif cfg["model"] == "cknrm":
        model = cknrm(cfg, embedding_init)
    elif cfg["model"] == "tk":
        model = tk(cfg, embedding_init)
    elif cfg["model"] == "edrm":
        model = edrm(cfg, embedding_init)
    elif cfg["model"] == "bert":
        model = bert(cfg)
    else:
        print("Model Error!")
        exit()
    model.to(device)

    # init optimizer and load dev_data
    if cfg["model"] != "bert":
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        dev_data = dev_dataloader(cfg, word2vec)
    else:
        optimizer_grouped_parameters = [{'params': [], 'weight_decay': 0.0}]
        param_optimizer = list(model.named_parameters())
        for n, p in param_optimizer:
            optimizer_grouped_parameters[0]['params'].append(p)
        m_optim = AdamW(optimizer_grouped_parameters, lr=5e-5)
        dev_data = bert_dev_dataloader(cfg, tokenizer)

    # load trained model
    if cfg["finetune"]:
        state_dict=torch.load(cfg["load_model"])
        model.load_state_dict(state_dict)

    # loss function
    crit = nn.MarginRankingLoss(margin=1, size_average=True)
    crit.to(device)

    if cfg["multi_gpu"]:
        policy = nn.DataParallel(policy)
        model = nn.DataParallel(model)
        crit = nn.DataParallel(crit)

    train(cfg, policy, p_optim, model, m_optim, crit, dev_data, device)

if __name__ == "__main__":
    main()
