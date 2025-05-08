from . import xlora_utils
from .xlora import (
    add_xlora_to_model,
    from_pretrained,
)
from .xlora_config import xLoRAConfig
from .xlora_attention_patch import patch_transformers_attention

# Apply the attention patch when the module is imported
patch_transformers_attention()