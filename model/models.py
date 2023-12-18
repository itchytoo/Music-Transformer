import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class Classifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.custom_linear = nn.Linear(self.pretrained_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]
        pooled_output = torch.mean(hidden_state, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.custom_linear(pooled_output)
        return logits