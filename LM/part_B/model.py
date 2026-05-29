# Add the class of your model only
# Here is where you define the architecture of your model using pytorch


from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel



class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, rank, alpha):
        super().__init__(config)
        #add layers to implement LoRA

        hidden_size = config.hidden_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha/rank

        #A
        self.q_lora_A = nn.Linear(hidden_size, rank)
        self.k_lora_A = nn.Linear(hidden_size, rank)
        self.v_lora_A = nn.Linear(hidden_size, rank)

        #B
        self.q_lora_B = nn.Linear(rank, hidden_size)
        self.k_lora_B = nn.Linear(rank, hidden_size)
        self.v_lora_B = nn.Linear(rank, hidden_size)

        nn.init.normal_(self.q_lora_A.weight, std=0.02)
        nn.init.normal_(self.k_lora_A.weight, std=0.02)
        nn.init.normal_(self.v_lora_A.weight, std=0.02)
        #B is 0 so initial step is a noop
        nn.init.zeros_(self.q_lora_B.weight)
        nn.init.zeros_(self.k_lora_B.weight)
        nn.init.zeros_(self.v_lora_B.weight)

        self.lora_layers = nn.ModuleList([
            self.q_lora_A,
            self.k_lora_A,
            self.v_lora_A,
            self.q_lora_B,
            self.k_lora_B,
            self.v_lora_B
        ])


    # edit the forward method to implement LoRa
    # from transformers 4.38.0
    # https://github.com/huggingface/transformers/blob/v4.38.0/src/transformers/models/gpt2/modeling_gpt2.py
    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[Tuple[torch.Tensor]] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        query_lora = self.q_lora_B(self.q_lora_A(hidden_states)) * self.scaling
        key_lora = self.k_lora_B(self.k_lora_A(hidden_states)) * self.scaling
        value_lora = self.v_lora_B(self.v_lora_A(hidden_states)) * self.scaling

        query = query + query_lora
        key = key + key_lora
        value = value + value_lora

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reorder_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
class GPT2_LoRA(GPT2LMHeadModel):
    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        #add custom attention blocks layers
        for block in self.transformer.h:
            #substitute block.attn with a new instance of CustomGPT2Attention
            attention = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            #keep the weights from block.attn and apply them to the new instance using .load_state_dict()
            attention.load_state_dict(block.attn.state_dict(), strict=False)
            block.attn = attention

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    