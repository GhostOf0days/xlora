import torch
import torch.nn.functional as F
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
            # Try the original implementation first, pass all args through kwargs
            # This avoids parameter conflicts with newer transformers versions
            all_kwargs = {
                'hidden_states': hidden_states,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'past_key_value': past_key_value,
                'output_attentions': output_attentions,
                'use_cache': use_cache,
                **kwargs
            }
            return original_forward(self, **all_kwargs)
        except RuntimeError as e:
            error_msg = str(e)
            # Only catch tensor dimension mismatch errors in scaled_dot_product_attention
            if "The expanded size of the tensor" in error_msg and "must match the existing size" in error_msg:
                print(f"Handling attention dimension mismatch: {error_msg}")
                
                # Extract dimensions from the error message
                # Example error: "The expanded size of the tensor (200) must match the existing size (100) at non-singleton dimension 3"
                import re
                expanded_size = int(re.search(r"The expanded size of the tensor \((\d+)\)", error_msg).group(1))
                existing_size = int(re.search(r"must match the existing size \((\d+)\)", error_msg).group(1))
                
                # Process inputs normally up to the point where scaled_dot_product_attention would be called
                bsz, q_len = hidden_states.shape[0], hidden_states.shape[1]
                
                # Standard LlamaAttention forward pass
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
                
                # This is the critical fix - handle dimension mismatch by resizing tensors
                if expanded_size > existing_size:
                    # Case when query is larger - resize key to match query
                    key_len = key_states.size(2)
                    key_states = torch.nn.functional.pad(
                        key_states, 
                        (0, expanded_size - existing_size, 0, 0, 0, 0, 0, 0),
                        "constant", 0
                    )
                    print(f"Padded key states from shape {existing_size} to {expanded_size}")
                else:
                    # Case when key is larger - resize query to match key
                    query_states = torch.nn.functional.pad(
                        query_states, 
                        (0, existing_size - expanded_size, 0, 0, 0, 0, 0, 0),
                        "constant", 0
                    )
                    print(f"Padded query states from shape {expanded_size} to {existing_size}")
                
                # Apply custom attention calculation
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(
                    torch.tensor(self.head_dim, dtype=query_states.dtype, device=query_states.device)
                )
                
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, value_states)
                
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                
                attn_output = self.o_proj(attn_output)
                
                return attn_output, attn_weights, past_key_value
            else:
                # Re-raise other errors
                raise
    
    # Apply the patch
    LlamaAttention.forward = patched_forward
    print("🔧 Patched LlamaAttention.forward to handle dimension mismatches in xLoRA") 
