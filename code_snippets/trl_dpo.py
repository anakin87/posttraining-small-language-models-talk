from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset("mlabonne/orpo-dpo-mix-40k")

cfg = DPOConfig(
    output_dir='./mymodel',
    max_length=1024,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                       
    learning_rate=5.0e-06,
    per_device_train_batch_size=1,
)

dpo_trainer = DPOTrainer(
    model="google/gemma-2-2b-it",
    args=cfg,
    train_dataset=dataset["train"],
)

dpo_trainer.train()