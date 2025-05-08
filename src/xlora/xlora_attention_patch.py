#!/usr/bin/env python3
"""
Test script for the xLoRA attention patch fix.
This script tests the attention dimension mismatch handling and evaluates the results with different adapters.
"""
import os
import sys
import torch
import argparse
from tqdm import tqdm
import numpy as np
from termcolor import colored

# Add the local xlora module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xlora-master', 'src'))
from xlora.xlora_attention_patch import patch_transformers_attention

# Apply the patch manually to ensure it's applied
patch_transformers_attention()

import xlora
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def print_success(msg):
    print(colored(f"✅ {msg}", "green"))

def print_failure(msg):
    print(colored(f"❌ {msg}", "red"))

def print_info(msg):
    print(colored(f"ℹ️ {msg}", "blue"))

def load_adapters():
    """Find all available adapters"""
    print_info("Looking for adapters in finetuned_models/")
    adapter_groups = {
        "8bits": ["finer_llama_3_1_8B_8bits_r8", "ner_llama_3_1_8b_8bits_r8", 
                 "headline_llama_3_1_8b_8bits_r8", "sentiment_llama_3_1_8b_8bits_r8"],
        "4bits": ["finer_llama_3_1_8B_4bits_r4", "ner_llama_3_1_8b_4bits_r4", 
                 "headline_llama_3_1_70b_4bits_r4", "sentiment_llama_3_1_8b_4bits_r4"]
    }
    
    valid_adapters = {}
    for precision, adapters in adapter_groups.items():
        valid_in_group = {}
        for adapter in adapters:
            path = f"finetuned_models/{adapter}"
            if os.path.exists(path):
                name = adapter.split("_")[0]  # Extract first part as name
                print_success(f"Found {precision} adapter: {name} at {path}")
                valid_in_group[name] = path
            else:
                print_failure(f"Missing {precision} adapter: {path} not found")
        
        if valid_in_group:
            print_info(f"Found {len(valid_in_group)} adapters in {precision} group")
            valid_adapters[precision] = valid_in_group
    
    return valid_adapters

def load_mixed_precision_model(precision="8bits"):
    """Load model with adapters of different precision or rank"""
    adapter_groups = load_adapters()
    
    if not adapter_groups:
        print_failure("No adapter groups found")
        return None, None
    
    if precision not in adapter_groups:
        print_failure(f"No adapters found for precision {precision}")
        precision = list(adapter_groups.keys())[0]
        print_info(f"Using {precision} adapters instead")
    
    adapters = adapter_groups[precision]
    
    if not adapters:
        print_failure(f"No {precision} adapters found")
        return None, None
    
    # Configure quantization
    if precision == "8bits":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    elif precision == "4bits":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
    
    # Load base model
    model_name = "meta-llama/Llama-3.1-8B"
    print_info(f"Loading base model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    
    # For xLoRA to work properly
    model.config.use_cache = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print_info("Creating xLoRA model with mixed adapters")
    
    # Create xLoRA config
    xlora_cfg = xlora.xLoRAConfig(
        hidden_size=model.config.hidden_size,
        base_model_id=model_name,
        adapters=adapters,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        top_k_lora=2,  # Use top-2 adapters
        use_trainable_adapters=False,
        enable_softmax=True,
        softmax_temperature=0.1
    )
    
    # Convert to xLoRA model
    print_info("Converting to xLoRA...")
    model = xlora.add_xlora_to_model(model=model, xlora_config=xlora_cfg, verbose=True)
    
    return model, tokenizer

def test_attention_patch(model, tokenizer):
    """Test the attention patch with various inputs"""
    if model is None or tokenizer is None:
        print_failure("Model or tokenizer is None, cannot run test")
        return False
    
    test_prompts = [
        # Short prompt with simple text
        "Analyze this financial statement: Revenue increased by 10% quarter-over-quarter.",
        
        # Medium length prompt with some financial text
        "Extract entities from this text: Microsoft Corporation (MSFT) reported earnings of $2.69 per share, beating analyst estimates of $2.53 per share.",
        
        # Longer prompt with more complex financial content
        """Generate a headline for this financial news article:
        The Federal Reserve announced today that it will maintain interest rates at their current level, citing concerns about inflation and employment figures. Analysts had expected a 25 basis point cut, but the central bank opted for a more cautious approach given recent economic data."""
    ]
    
    all_successful = True
    
    for i, prompt in enumerate(test_prompts):
        print_info(f"\nTesting prompt {i+1} ({len(prompt)} chars)")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Test forward pass
        try:
            print_info("Testing forward pass...")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            print_success("Forward pass succeeded")
            
            # Get adapter weights
            if hasattr(model, "get_latest_scalings"):
                scalings = model.get_latest_scalings()
                if scalings is not None:
                    weights = scalings[0, 0, 0].float().cpu().numpy()
                    print_info("Adapter weights:")
                    for i, adapter_name in enumerate(model.base_model.active_adapters):
                        print(f"  - {adapter_name}: {weights[i]:.3f}")
            
            # Test generation
            print_info("Testing generation...")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            response = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print_info(f"Generated: {response.strip()}")
            print_success("Generation succeeded")
        
        except Exception as e:
            print_failure(f"Error during test: {str(e)}")
            import traceback
            traceback.print_exc()
            all_successful = False
            continue
    
    if all_successful:
        print_success("\nAll tests passed successfully!")
    else:
        print_failure("\nSome tests failed")
    
    return all_successful

def main():
    parser = argparse.ArgumentParser(description="Test xLoRA attention patch fix")
    parser.add_argument("--precision", type=str, default="8bits", choices=["4bits", "8bits"], 
                        help="Precision of adapters to test with")
    args = parser.parse_args()
    
    print_info(f"Testing attention patch with {args.precision} adapters")
    model, tokenizer = load_mixed_precision_model(precision=args.precision)
    
    success = test_attention_patch(model, tokenizer)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 
