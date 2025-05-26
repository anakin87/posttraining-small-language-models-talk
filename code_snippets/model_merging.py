yaml_config = """
slices:
  - sources:
      - model: AIDC-ai-business/Marcoroni-7B-v3
        layer_range: [0, 32]
      - model: EmbeddedLLM/Mistral-7B-Merge-14-v0.1
        layer_range: [0, 32]
merge_method: slerp
base_model: AIDC-ai-business/Marcoroni-7B-v3
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
"""

with open('config.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

! mergekit-yaml config.yaml merge --copy-tokenizer --allow-crimes --out-shard-size 1B --lazy-unpickle    