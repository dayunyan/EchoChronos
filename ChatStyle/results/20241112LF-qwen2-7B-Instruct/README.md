---
library_name: peft
license: other
base_model: /home/zjj/xjd/huawei-ict-2024/ChatStyle/.mindnlp/model/Qwen/Qwen2-7B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train_2024-11-11-20-02-57
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2024-11-11-20-02-57

This model is a fine-tuned version of [/home/zjj/xjd/huawei-ict-2024/ChatStyle/.mindnlp/model/Qwen/Qwen2-7B-Instruct](https://huggingface.co//home/zjj/xjd/huawei-ict-2024/ChatStyle/.mindnlp/model/Qwen/Qwen2-7B-Instruct) on the chatstyle dataset.
It achieves the following results on the evaluation set:
- Loss: 1.8947

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 4
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.9615        | 0.4182 | 100  | 1.9039          |
| 1.856         | 0.8364 | 200  | 1.8871          |
| 1.9986        | 1.2546 | 300  | 1.8832          |
| 1.8464        | 1.6728 | 400  | 1.8843          |
| 1.7402        | 2.0910 | 500  | 1.8910          |
| 1.6629        | 2.5091 | 600  | 1.8945          |
| 1.8994        | 2.9273 | 700  | 1.8943          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu124
- Datasets 3.1.0
- Tokenizers 0.20.3