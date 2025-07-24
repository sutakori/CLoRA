# CLoRA
Code for ACL 2025 paper "Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on Large Language Models"

The file clora.py provides a wrapper for CLoRA that compatible with huggingface [PEFT](https://github.com/huggingface/peft) library. 

## Usage
Warp the PEFT model with CLoRAWrapper.

```python
from peft import get_peft_model
from .clora import CLoraWrapper

peft_model = get_peft_model(model, peft_config)
peft_model = CLoraWrapper(peft_model, adapter_name='default', k=512)
```
