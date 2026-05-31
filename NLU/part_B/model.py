import torch
import torch.nn as nn
from transformers import AutoModel

GPT2_PATH = 'openai-community/gpt2'
BERT_PATH = 'bert-base-uncased'

class HuggingFaceModel(nn.Module):
    def __init__(self, model_name, n_slots, n_intents, dropout=0.1):
        super().__init__()

        #load pretrained model, 0=gpt2, 1=bert
        self.model_type = "gpt2" if model_name==GPT2_PATH else "bert"
        if model_name==GPT2_PATH:
            self.backbone = AutoModel.from_pretrained(GPT2_PATH)
        else:
            self.backbone = AutoModel.from_pretrained(BERT_PATH)

        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.backbone.config.hidden_size

        #the two learning heads
        self.slot_out = nn.Linear(hidden_size, n_slots)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # last hidden state shape (batch size, sequence length, hidden size)
        sequence_output = self.dropout(outputs.last_hidden_state)

        # INTENT CLASSIFICATION
        if self.model_type == "gpt2":
            #Calculate the index of the last non-padding token for each item in the batch
            # sum(dim=1) gives the length of each sequence
            sequence_lengths = attention_mask.sum(dim=1).long()
            
            #Select the hidden state at the last valid index (length - 1)
            batch_size = sequence_output.shape[0]
            intent_logits = self.intent_out(sequence_output[torch.arange(batch_size), sequence_lengths - 1])
        else:
            # BERT uses the first token ([CLS]) of the sequence
            intent_logits = self.intent_out(sequence_output[:, 0, :])
        
        # SLOT FILLING
        slot_logits = self.slot_out(sequence_output)
        return slot_logits, intent_logits