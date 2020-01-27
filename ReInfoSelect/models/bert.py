import torch.nn as nn
from transformers import *

class BertForRanking(nn.Module):
    def __init__(self, cfg):
        super(BertForRanking, self).__init__()
        config = BertConfig.from_pretrained(cfg["config"])
        self.bert = BertModel.from_pretrained(cfg["model"], config=config)
        self.bert.train()

        feature_dim = config.hidden_size
        if cfg["score_feature"]:
            feature_dim = config.hidden_size + 1
        self.dense = nn.Linear(feature_dim, 1)

    def forward(self, inst, tok, mask, score_feature=None):
        output = self.bert(inst, token_type_ids=tok, attention_mask=mask)
        if score_feature is not None:
            logits = torch.cat([output[1], score_feature.unsqueeze(1)], 1)
        score = torch.squeeze(self.dense(output[1]), dim=1)
        return score, output[1]
