import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, num_labels):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('../bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        feature = self.bert(input_ids, attention_mask).last_hidden_state[:, 0, :]
        #feature = self.bert(input_ids,attention_mask,output_hidden_states =True).hidden_states[-2][:,0,:]
        logits = self.fc(self.dropout(feature))
        return logits,feature