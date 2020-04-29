import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel

class BertForRanking(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForRanking, self).__init__(config)

        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, raw_score=None):
        _, features = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        score = self.dense(features).squeeze(-1)
        if raw_score is not None:
            features = torch.cat([features, raw_score.unsqueeze(1)], 1)
        return score, features
