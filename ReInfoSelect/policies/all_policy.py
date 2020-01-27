import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

import texar as tx

def kernal_mus(n_kernels):
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu

def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class all_policy(nn.Module):
    def __init__(self, cfg, embedding_init=None):
        super(all_policy, self).__init__()
        tensor_mu = torch.FloatTensor(kernal_mus(cfg["n_kernels"]))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(cfg["n_kernels"]))
        if torch.cuda.is_available():
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, cfg["n_kernels"])
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, cfg["n_kernels"])

        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        if embedding_init is not None:
            em = torch.tensor(embedding_init, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(em)
            self.embedding.weight.requires_grad = True

        self.q_clas = tx.modules.Conv1DClassifier(cfg["channels"], cfg["embedding_dim"], {'num_classes': 0})
        self.d_clas = tx.modules.Conv1DClassifier(cfg["channels"], cfg["embedding_dim"], {'num_classes': 0})

        self.q_actor = nn.Linear(256, 2)
        self.d_actor = nn.Linear(256, 2)
        self.qd_actor = nn.Linear(cfg["n_kernels"], 2)

        self.actor = nn.Linear(3, 1)
    def forward(self, query_idx, doc_idx, query_len, doc_len):
        query_embed = self.embedding(query_idx)
        q_logits, q_pred = self.q_clas(query_embed, query_len)
        q_logits = F.relu(q_logits)
 
        doc_embed = self.embedding(doc_idx[:, :20])
        d_logits, d_pred = self.d_clas(doc_embed, doc_len)
        d_logits = F.relu(d_logits)

        query_mask = self.create_mask_like(query_len, query_embed)
        doc_mask = self.create_mask_like(doc_len, doc_embed)
        qd_logits = self.kernel_pooling(query_embed, doc_embed, query_mask, doc_mask)
        qd_logits = F.relu(qd_logits)

        q_probs = F.softmax(self.q_actor(q_logits), dim=-1).unsqueeze(2)
        d_probs = F.softmax(self.d_actor(d_logits), dim=-1).unsqueeze(2)
        qd_probs = F.softmax(self.qd_actor(qd_logits), dim=-1).unsqueeze(2)

        all_probs = torch.cat([q_probs, d_probs, qd_probs], 2)
        probs = F.softmax(self.actor(all_probs).squeeze(2), dim=1)
        return probs

    def create_mask_like(self, lengths, like):
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        return mask

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q.type_as(self.sigma) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def kernel_pooling(self, q_embed, d_embed, mask_q, mask_d):
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)

        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        return log_pooling_sum
