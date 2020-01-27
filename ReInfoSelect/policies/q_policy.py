import torch
from torch import nn
import torch.nn.functional as F

import texar as tx

class q_policy(nn.Module):
    def __init__(self, cfg, embedding_init=None):
        super(q_policy, self).__init__()
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        if embedding_init is not None:
            emb = torch.tensor(embedding_init, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(emb)
            self.embedding.weight.requires_grad = True

        self.classifier = tx.modules.Conv1DClassifier(cfg["channels"], cfg["embedding_dim"], {'num_classes': 0})

        self.actor = nn.Linear(256, 2)
    def forward(self, query_idx, query_len):
        query_embed = self.embedding(query_idx)
        logits, pred = self.classifier(query_embed, query_len)

        logits = F.relu(logits)
        probs = F.softmax(self.actor(logits), dim=1)
        return probs
