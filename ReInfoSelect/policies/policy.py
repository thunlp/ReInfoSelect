import torch
from torch import nn
import torch.nn.functional as F

from models import cknrm

class all_policy(nn.Module):
    def __init__(self, args, embedding_init=None):
        super(all_policy, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)
        if embedding_init is not None:
            em = torch.tensor(embedding_init, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(em)
            self.embedding.weight.requires_grad = True

        kernels = [1, 2, 3]
        self.q_clas = nn.ModuleList([nn.Conv2d(1, 100, (k, args.embed_dim)) for k in kernels])
        self.d_clas = nn.ModuleList([nn.Conv2d(1, 100, (k, args.embed_dim)) for k in kernels])
        self.qd_clas = cknrm(args, embedding_init)

        self.q_actor = nn.Linear(len(kernels)*100, 2)
        self.d_actor = nn.Linear(len(kernels)*100, 2)
        self.qd_actor = nn.Linear(args.n_kernels*9, 2)

        self.actor = nn.Linear(3, 1)

    def forward(self, query_idx, doc_idx, query_len, doc_len):
        query_embed = self.embedding(query_idx)
        query_embed = query_embed.unsqueeze(1)
        q_logits = [F.relu(conv(query_embed)).squeeze(3) for conv in self.q_clas]
        q_logits = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in q_logits]
        q_logits = torch.cat(q_logits, 1)
 
        doc_embed = self.embedding(doc_idx)
        doc_embed = doc_embed.unsqueeze(1)
        d_logits = [F.relu(conv(doc_embed)).squeeze(3) for conv in self.d_clas]
        d_logits = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in d_logits]
        d_logits = torch.cat(d_logits, 1)

        _, qd_logits = self.qd_clas(query_idx, doc_idx, query_len, doc_len)

        q_probs = self.q_actor(q_logits).unsqueeze(2)
        d_probs = self.d_actor(d_logits).unsqueeze(2)
        qd_probs = self.qd_actor(qd_logits).unsqueeze(2)

        all_probs = torch.cat([q_probs, d_probs, qd_probs], 2)
        probs = F.softmax(self.actor(all_probs).squeeze(2), dim=1)
        return probs
