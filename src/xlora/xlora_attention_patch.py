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
                
                # Ensure correct return format for generation
                if use_cache:
                    # For generation, maintain proper format with past_key_value
                    return attn_output, attn_weights if output_attentions else None, past_key_value
                else:
                    # For normal forward pass
                    return attn_output, attn_weights if output_attentions else None
