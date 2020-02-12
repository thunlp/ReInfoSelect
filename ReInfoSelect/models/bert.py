import torch.nn as nn
from transformers import *

class BertForRanking(nn.Module):
    def __init__(self):
        super(BertForRanking, self).__init__()
        model = model_class.from_pretrained('bert-base-uncased')

        feature_dim = 768
        self.dense = nn.Linear(feature_dim, 1)
        self.dense_p = nn.Linear(feature_dim + 1, 1)

    def forward(self, inst, tok, mask, raw_score=None):
        output = self.bert(inst, token_type_ids=tok, attention_mask=mask)
        if score_feature is not None:
            logits = torch.cat([output[1], raw_score.unsqueeze(1)], 1)
            score = self.dense_p(output[1]).squeeze(-1)
        else:
            score = self.dense(output[1]).squeeze(-1)
        return score, output[1]
