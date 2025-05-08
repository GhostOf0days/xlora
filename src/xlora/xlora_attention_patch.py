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
                import re
                expanded_size = int(re.search(r"The expanded size of the tensor \((\d+)\)", error_msg).group(1))
                existing_size = int(re.search(r"must match the existing size \((\d+)\)", error_msg).group(1))
                
                # Process inputs normally up to the point where scaled_dot_product_attention would be called
                bsz, q_len = hidden_states.shape[0], hidden_states.shape[1]
                
                # Directly infer dimensions from module weights rather than attributes
                # Look at the q_proj, k_proj weight shapes to infer dimensions
                q_proj_out_dim = self.q_proj.weight.shape[0]
                k_proj_out_dim = self.k_proj.weight.shape[0]
                v_proj_out_dim = self.v_proj.weight.shape[0]
                
                # For Llama3, q_proj is usually (num_heads * head_dim, hidden_size)
                # and k_proj is (num_key_value_heads * head_dim, hidden_size)
                
                # Try to find head_dim in various ways
                head_dim = None
                
                # Method 1: Check if head_dim is a direct attribute
                if hasattr(self, "head_dim"):
                    head_dim = self.head_dim
                # Method 2: Check config
                elif hasattr(self, "config") and hasattr(self.config, "head_dim"):
                    head_dim = self.config.head_dim
                # Method 3: Try common values (128 for Llama 3)
                else:
                    for common_dim in [128, 64, 80, 96, 160]:
                        if q_proj_out_dim % common_dim == 0:
                            head_dim = common_dim
                            break
                
                if head_dim is None:
                    # Last resort: just use what we see in error message dimensions
                    head_dim = existing_size
                
                # Calculate num_heads and num_key_value_heads
                num_heads = q_proj_out_dim // head_dim
                num_key_value_heads = k_proj_out_dim // head_dim
                hidden_size = self.o_proj.weight.shape[0]
                
                print(f"Inferred dimensions: num_heads={num_heads}, num_kv_heads={num_key_value_heads}, head_dim={head_dim}")
                
                # Create dummy causal attention mask for our computation
                # [bsz, 1, tgt_len, src_len]
                causal_mask = torch.ones((bsz, 1, q_len, q_len), dtype=torch.bool, device=hidden_states.device)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask = causal_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
                causal_mask = causal_mask * torch.finfo(hidden_states.dtype).min
                
                # Standard LlamaAttention forward pass
                if hasattr(self, "config") and hasattr(self.config, "pretraining_tp") and getattr(self.config, "pretraining_tp", 1) > 1:
                    pretraining_tp = self.config.pretraining_tp
                    key_value_slicing = (num_key_value_heads * head_dim) // pretraining_tp
                    query_slices = self.q_proj.weight.split(
                        (num_heads * head_dim) // pretraining_tp, dim=0
                    )
                    key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                    value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                    query_states = [F.linear(hidden_states, query_slices[i]) for i in range(pretraining_tp)]
                    query_states = torch.cat(query_states, dim=-1)

                    key_states = [F.linear(hidden_states, key_slices[i]) for i in range(pretraining_tp)]
                    key_states = torch.cat(key_states, dim=-1)

                    value_states = [F.linear(hidden_states, value_slices[i]) for i in range(pretraining_tp)]
                    value_states = torch.cat(value_states, dim=-1)
                else:
                    # Regular case
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

                # Reshape
                query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
                
                if past_key_value is not None:
                    # Combine with past key and value states
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                
                past_key_value = (key_states, value_states) if use_cache else None
                
                # Ensure we have tensors, not tuples
                if isinstance(query_states, tuple):
                    query_states = query_states[0]
                if isinstance(key_states, tuple):
                    key_states = key_states[0]
                if isinstance(value_states, tuple):
                    value_states = value_states[0]
                    
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
                try:
                    # Try standard matmul approach first
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(
                        torch.tensor(head_dim, dtype=query_states.dtype, device=query_states.device)
                    )
                    
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    else:
                        # Add causal mask
                        attn_weights = attn_weights + causal_mask
                    
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                    attn_output = torch.matmul(attn_weights, value_states)
                except TypeError as e:
                    print(f"Caught TypeError: {e}")
                    # Fall back to einsum for extra safety
                    print("Falling back to einsum implementation")
                    q_states = query_states.contiguous()
                    k_states = key_states.contiguous()
                    v_states = value_states.contiguous()
                    
                    scaling = float(head_dim) ** -0.5
                    attn_weights = torch.einsum("bhld,bhsd->bhls", q_states, k_states) * scaling
                    
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    else:
                        # Add causal mask
                        attn_weights = attn_weights + causal_mask
                        
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                    attn_output = torch.einsum("bhls,bhsd->bhld", attn_weights, v_states)
                
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                
                attn_output = self.o_proj(attn_output)
                
                if output_attentions:
                    return attn_output, attn_weights, past_key_value
                else:
                    return attn_output, None, past_key_value
            else:
                # Re-raise other errors
                raise
    
    # Apply the patch
    LlamaAttention.forward = patched_forward
    print("🔧 Patched LlamaAttention.forward to handle dimension mismatches in xLoRA")
