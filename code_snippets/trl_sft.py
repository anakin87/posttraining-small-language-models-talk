from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("efederici/capybara-claude-15k-ita")

cfg = SFTConfig(
    output_dir='./mymodel',
    max_seq_length=2048,
    packing=True,    
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.2,                       
    learning_rate=5.0e-06,
    per_device_train_batch_size=8,
)

sft_trainer = SFTTrainer(
    model="microsoft/Phi-3.5-mini-instruct",
    args=cfg,
    train_dataset=dataset["train"],
)

sft_trainer.train()