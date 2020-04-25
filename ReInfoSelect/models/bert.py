import torch
import torch.nn as nn
from transformers import *

class BertForRanking(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForRanking, self).__init__(config)

        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        score = self.dense(output[1])[:, 1]
        return score, output[1]
