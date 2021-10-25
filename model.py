from util import *

        
class BertForModel(RobertaForMaskedLM):
    def __init__(self,config, n_labels):
        super(BertForModel, self).__init__(config)
        self.n_labels = n_labels
        self.roberta = RobertaModel(config)
        # self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=True)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, n_labels)

    def forward(self, input_ids = None, attention_mask=None, labels = None, mode = None, centroids = None, labeled = False, feature_ext = False):
        encoded_layer_12 = self.roberta(input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
        # pooled_output, (hn, cn) = self.lstm(encoded_layer_12)
        pooled_output = self.dense(encoded_layer_12.mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)        
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return pooled_output, logits
