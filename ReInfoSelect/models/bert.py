import torch
import torch.nn as nn

from transformers import AutoModel

class Bert(nn.Module):
    def __init__(self, pretrained, enc_dim):
        super(Bert, self).__init__()
        self._model = AutoModel.from_pretrained(pretrained)
        self._dense = nn.Linear(enc_dim, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, raw_score=None):
        _, features = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        score = self.dense(features).squeeze(-1)
        if raw_score is not None:
            features = torch.cat([features, raw_score.unsqueeze(1)], 1)
        return score, features
