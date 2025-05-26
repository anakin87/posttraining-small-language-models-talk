# Post-Training Small Language Models: the adventures of a practitioner

Material for the homonymous talk.

<details><summary>📝 Abstract</summary>
In 2025, AI is still evolving rapidly. While closed LLMs are continuously improving, open Small Language Models are emerging as powerful alternatives for specific use cases, consuming only a fraction of the resources.

Working in AI engineering, I often find it refreshing to step away from orchestration and get hands-on with fine-tuning, customizing, and optimizing Small Models. In this talk, I'll share my journey working with Post-Training Small Language Models, full of joys, frustrations, and many lessons learned.

We'll see together:

- How generative Language Models are trained and how we can further customize them
- Tips for collecting and generating data for fine-tuning
- Instruction Fine-Tuning and Preference Tuning (DPO)
- Key training libraries, with a focus on Hugging Face TRL.
- Low-resource fine-tuning methods (QLoRA, Spectrum).
- A look at quantization and model merging.

By the end, you'll learn how to customize Small Language Models for your needs and potentially run them on your smartphone.

I'll also share practical examples from my experience improving open models for the Italian language.

<b>UPDATE</b>
This world changed a bit since when I proposed the talk, so I added a section about Reasoning models and GRPO!
</details>


## 📚 Resources and code ‍💻

- [🌱 Intro](#-intro)
- [👣 Common Post Training steps](#-common-post-training-steps)
- [⚙️💰 Memory-efficient training](#️-memory-efficient-training)
- [🧩 Model merging](#-model-merging)
- [🧠💭 Reasoning models and GRPO](#-reasoning-models-and-grpo)
- [💰 Quantization](#-quantization)
- [📱 Small Language Models on a phone](#-small-language-models-on-a-phone)

### 🌱 Intro

#### Evaluation ⚖️
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): common framework for evaluating Language Models.
- [🤗 YourBench](https://github.com/huggingface/lm-bench): open-source framework for generating domain-specific benchmarks.

#### Choosing the model to train
- How to approach post-training for AI applications by Nathan Lambert: [slides](https://docs.google.com/presentation/d/1LWHbtz74GwKSGYZKyBVUtcyvp8lgYOi5EVpMnVDXBPs); [video](https://www.youtube.com/watch?v=grpc-Wyy-Zg).


### 👣 Common Post Training steps

#### Supervised Fine-Tuning (SFT)

##### SFT with TRL
- [Code snippet](./code_snippets/trl_sft.py)
- [TRL docs](https://huggingface.co/docs/trl/sft_trainer)

##### SFT projects
- [Fine-Tune Your Own Llama 2 Model in a Colab Notebook by Maxime Labonne](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html)
- [Fine-Tune Phi 3.5 mini on Italian](https://huggingface.co/blog/anakin87/spectrum)

#### Preference Alignment
- [RLHF/PPO: InstructGPT paper](https://arxiv.org/abs/2203.02155)
- [DPO paper](https://arxiv.org/abs/2305.18290)

##### Direct Preference Optimization (DPO) with TRL
- [Code snippet](./code_snippets/trl_dpo.py)
- [TRL docs](https://huggingface.co/docs/trl/dpo_trainer)

##### DPO projects
- DPO only: [Fine-tune Mistral-7b with Direct Preference Optimization by Maxime Labonne](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)
- DPO + SFT: [Post-training Gemma for Italian and beyond](http://kaggle.com/code/anakin87/post-training-gemma-for-italian-and-beyond)

#### Supervised Fine-Tuning vs Preference Alignment
- [Disentangling Post-training performance elicitation from data by Mohit Raghavendra](https://mohit-raghavendra.notion.site/Disentangling-Post-training-performance-elicitation-from-data-1a5db7f2a34480e18010d689a1f46f74)

#### Tips from practice
- [Distilabel](https://distilabel.argilla.io/): unmaintained project but good for inspiration on several techniques to generate synthetic data.
- Setting `max_seq_length`, `max_prompt_length`, and `max_length`: good explanation on this article by [Philipp Schmid](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl).

### ⚙️💰 Memory-efficient training

#### LoRA and QLoRA
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [QLoRA with TRL: code snippet](./code_snippets/trl_sft_qlora.py)

#### Spectrum
- [Spectrum: Targeted Training on Signal to Noise Ratio](https://arxiv.org/abs/2406.06623)
- [Spectrum with TRL: code snippet](./code_snippets/trl_sft_spectrum.py)

#### Projects on memory-efficient training
- [QLoRA tutorial by Philipp Schmid](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl)
- [QLoRA on Gemma from Google documentation](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)
- [Selective fine-tuning of Language Models with Spectrum - tutorial](https://huggingface.co/blog/anakin87/spectrum)

### 🧩 Model merging
- [Model merging with Mergekit: code snippet](./code_snippets/model_merging.py)
- [Merge Large Language Models with MergeKit: blogpost by Maxime Labonne](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html)

### 🧠💭 Reasoning models and GRPO
- Series of articles on reasoning models by Sebastian Raschka: [1](https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html),
[2](https://sebastianraschka.com/blog/2025/first-look-at-reasoning-from-scratch.html), [3](https://sebastianraschka.com/blog/2025/the-state-of-reinforcement-learning-for-llm-reasoning.html).
- [Build Reasoning models: chapter from Hugging Face LLM course](https://huggingface.co/learn/llm-course/en/chapter12)
- [GRPO with TRL: docs](https://huggingface.co/docs/trl/grpo_trainer)

#### GRPO projects
- [GRPO Llama-1B (GSM8K): gist by William Brown](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)
- [Qwen Scheduler GRPO: detailed walkthrough on training a reasoning model to solve a scheduling problem](http://hf.co/blog/anakin87/qwen-scheduler-grpo)

### 📱 Small Language Models on a phone

#### GGUF
- [GGUF My Repo](http://hf.co/spaces/ggml-org/gguf-my-repo): Hugging Face space to convert your model to GGUF format.
- [llama.cpp script for conversion](github.com/ggml-org/llama.cpp/tree/master/tools/quantize).