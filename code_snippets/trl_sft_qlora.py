from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

dataset = load_dataset(...)

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "..." 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

 
# LoRA config
peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=64,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
 
cfg = SFTConfig(...)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    args=cfg,
    train_dataset=dataset["train"],
)
sft_trainer.train()