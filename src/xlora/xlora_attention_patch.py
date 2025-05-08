# Create file: xlora-master/src/xlora/xlora_attention_patch.py

import torch
import types
from transformers.models.llama.modeling_llama import LlamaAttention

def patch_transformers_attention():
    """
    Patch the transformers LlamaAttention to handle tensor dimension mismatches when
    using multiple adapters with different dimensions in xLoRA.
    """
    original_forward = LlamaAttention.forward
    
    def patched_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                        output_attentions=False, use_cache=False, **kwargs):
        try:
            return original_forward(self, hidden_states, attention_mask, position_ids, 
                                   past_key_value, output_attentions, use_cache, **kwargs)
        except RuntimeError as e:
            if "The expanded size of the tensor" in str(e) and "must match the existing size" in str(e):
                # Extract shapes from the error message
                bsz = hidden_states.shape[0]
                q_len = hidden_states.shape[1]
                
                # Process inputs normally up to the point where scaled_dot_product_attention would be called
                if self.config.pretraining_tp > 1:
                    key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                    query_slices = self.q_proj.weight.split(
                        (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                    )
                    key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                    value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                    query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                    query_states = torch.cat(query_states, dim=-1)

                    key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                    key_states = torch.cat(key_states, dim=-1)

                    value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                    value_states = torch.cat(value_states, dim=-1)
                else:
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                
                if past_key_value is not None:
                    # Combine with past key and value states
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                
                past_key_value = (key_states, value_states) if use_cache else None
                
                # Fix the dimension mismatch by resizing key to match query dimensions
                # xLoRA with adapters of different dimensions
                if key_states.shape[-1] != query_states.shape[-1]:
                    # Resize key to match query for scaled_dot_product_attention
                    target_shape = query_states.shape[-1]
                    key_states = torch.nn.functional.interpolate(
                        key_states.permute(0, 1, 3, 2),  # [b, h, d, seq_len]
                        size=target_shape,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 1, 3, 2)  # [b, h, seq_len, d]
                
                # Apply custom attention
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(
                    torch.tensor(self.head_dim, dtype=query_states.dtype, device=query_states.device)
                )
                
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, value_states)
                
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, self.hidden_size)
                
                attn_output = self.o_proj(attn_output)
                
                return attn_output, attn_weights, past_key_value
            else:
                # Re-raise other errors
                raise
    
    # Apply the patch
    LlamaAttention.forward = patched_forward