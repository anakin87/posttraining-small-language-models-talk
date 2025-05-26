from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import re

# 1. USE SPECTRUM TO IDENTIFY PARAMETERS TO TRAIN
# git clone https://github.com/cognitivecomputations/spectrum.git && cd spectrum && pip install -r requirements.txt
# python spectrum.py --model-name <insert local or HF repo here> --top-percent <top % of snr ratios to target>

with open("snr_results.yaml", "r") as fin:
    yaml_parameters = fin.read()

unfrozen_parameters = []
for line in yaml_parameters.splitlines():
  if line.startswith("- "):
    unfrozen_parameters.append(line.split("- ")[1])

dataset = load_dataset(...)

model_id = "..." 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. FREEZE ALL PARAMETERS EXCEPT THOSE IDENTIFIED BY SPECTRUM
def freeze_and_unfreeze_parameters(model, unfrozen_parameters):
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze Spectrum parameters
    for name, param in model.named_parameters():
        if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
            param.requires_grad = True

freeze_and_unfreeze_parameters(model, unfrozen_parameters)
 
cfg = SFTConfig(...)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=cfg,
    train_dataset=dataset["train"],
)
sft_trainer.train()