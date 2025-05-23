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
                hidden_size = self.o_proj.weight.shape[0]
                
                # For Llama3, q_proj is usually (num_heads * head_dim, hidden_size)
                # and k_proj is (num_key_value_heads * head_dim, hidden_size)
                
                # Try to find head_dim in various ways
                head_dim = None
                
                # Method 1: Check if head_dim is a direct attribute (safest approach)
                if hasattr(self, "head_dim"):
                    head_dim = self.head_dim
                # Method 2: Check config
                elif hasattr(self, "config") and hasattr(self.config, "head_dim"):
                    head_dim = self.config.head_dim
                # Method 3: Try common values (128 for Llama 3)
                else:
                    for common_dim in [128, 64, 80, 96, 160]:
                        if q_proj_out_dim % common_dim == 0 and k_proj_out_dim % common_dim == 0:
                            head_dim = common_dim
                            break
                
                if head_dim is None:
                    # Last resort: just use what we see in error message dimensions
                    head_dim = existing_size
                
                # Calculate num_heads and num_key_value_heads
                num_heads = q_proj_out_dim // head_dim
                num_key_value_heads = k_proj_out_dim // head_dim
                
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
                
                # Ensure we have tensors, not tuples
                if isinstance(query_states, tuple):
                    query_states = query_states[0]
                if isinstance(key_states, tuple):
                    key_states = key_states[0]
                if isinstance(value_states, tuple):
                    value_states = value_states[0]
                
                # Store the original head dimension for later
                original_head_dim = head_dim
                
                # Proper past_key_value handling with type checking
                if past_key_value is not None:
                    try:
                        # Extract past key and value tensors, handling nested tuples
                        def get_tensor(x, idx=0):
                            if x is None:
                                return None
                            if isinstance(x, torch.Tensor):
                                return x
                            elif isinstance(x, tuple) and len(x) > idx:
                                return get_tensor(x[idx])
                            return None
                        
                        past_key = get_tensor(past_key_value, 0)
                        past_value = get_tensor(past_key_value, 1)
                        
                        # Only concatenate if both tensors are available
                        if past_key is not None and isinstance(past_key, torch.Tensor):
                            print(f"Concatenating past_key shape {past_key.shape} with key_states shape {key_states.shape}")
                            key_states = torch.cat([past_key, key_states], dim=2)
                        
                        if past_value is not None and isinstance(past_value, torch.Tensor):
                            value_states = torch.cat([past_value, value_states], dim=2)
                    except Exception as e:
                        print(f"Error handling past_key_value: {e}, type={type(past_key_value)}")
                        # Continue without using past_key_value
                
                # Make sure we have tensors at this point
                if not isinstance(key_states, torch.Tensor):
                    print(f"Converting key_states type {type(key_states)} to tensor")
                    key_states = torch.tensor(key_states) if hasattr(key_states, "__array__") else key_states[0]
                    
                if not isinstance(value_states, torch.Tensor):
                    print(f"Converting value_states type {type(value_states)} to tensor")
                    value_states = torch.tensor(value_states) if hasattr(value_states, "__array__") else value_states[0]
                    
                past_key_value = (key_states, value_states) if use_cache else None
                
                # Handle grouped query attention (GQA) where num_heads > num_key_value_heads
                if num_heads > num_key_value_heads:
                    # Repeat key and value states to match query heads
                    # For GQA: Each key/value head is used for multiple query heads
                    head_repeat_factor = num_heads // num_key_value_heads
                    
                    if head_repeat_factor > 1:
                        # [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
                        key_states = key_states.repeat_interleave(head_repeat_factor, dim=1)
                        value_states = value_states.repeat_interleave(head_repeat_factor, dim=1)
                        print(f"Repeated key/value states {head_repeat_factor}x to match query head count ({num_key_value_heads} -> {num_heads})")
                
                # Handle different head dimensions between query and key states (important for xLoRA with different ranks)
                # Instead of padding, we'll resize to a fixed size to ensure consistency
                query_head_dim = query_states.size(-1)
                key_head_dim = key_states.size(-1)
                
                if query_head_dim != key_head_dim:
                    print(f"Fixing head dimension mismatch: query={query_head_dim}, key={key_head_dim}")
                    
                    # Resize both query and key to the original head_dim (which should be 128 for Llama-3)
                    if query_head_dim != original_head_dim:
                        # Truncate or pad query head dimension to match original_head_dim
                        if query_head_dim > original_head_dim:
                            query_states = query_states[..., :original_head_dim]
                            print(f"Truncated query head dim from {query_head_dim} to {original_head_dim}")
                        else:
                            query_states = F.pad(query_states, (0, original_head_dim - query_head_dim), "constant", 0)
                            print(f"Padded query head dim from {query_head_dim} to {original_head_dim}")
                    
                    if key_head_dim != original_head_dim:
                        # Truncate or pad key head dimension to match original_head_dim
                        if key_head_dim > original_head_dim:
                            key_states = key_states[..., :original_head_dim]
                            value_states = value_states[..., :original_head_dim]
                            print(f"Truncated key/value head dim from {key_head_dim} to {original_head_dim}")
                        else:
                            key_states = F.pad(key_states, (0, original_head_dim - key_head_dim), "constant", 0)
                            value_states = F.pad(value_states, (0, original_head_dim - key_head_dim), "constant", 0)
                            print(f"Padded key/value head dim from {key_head_dim} to {original_head_dim}")
                
                # Apply custom attention calculation
                try:
                    # Try standard matmul approach first
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(
                        torch.tensor(original_head_dim, dtype=query_states.dtype, device=query_states.device)
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
                    
                    scaling = float(original_head_dim) ** -0.5
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
                
                # The caller expects exactly 2 return values: hidden_states, self_attn_weights
                # Store past_key_value in a way that doesn't affect the unpacking
                if use_cache:
                    # The LlamaDecoderLayer.forward expects only two return values
                    # So we need to wrap past_key_value with attn_weights in a single object
                    attn_weights_with_cache = (attn_weights if output_attentions else None, past_key_value)
                    return attn_output, attn_weights_with_cache
                else:
                    # Standard return for normal forward pass
                    return attn_output, attn_weights if output_attentions else None
            else:
                # Re-raise other errors
                raise
    
    # Apply the patch
    LlamaAttention.forward = patched_forward
    print("🔧 Patched LlamaAttention.forward to handle dimension mismatches in xLoRA") 
