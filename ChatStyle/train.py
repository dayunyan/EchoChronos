import json
import os
import mindspore as ms
from mindspore import context
from mindspore import nn
from mindnlp.core import optim
from mindnlp.transformers.optimization import get_polynomial_decay_schedule_with_warmup
from mindnlp import dataset as ds
from mindspore.nn.learning_rate_schedule import PolynomialDecayLR
from mindspore.communication import init
from mindspore.amp import auto_mixed_precision
from tqdm import tqdm
import numpy as np
import argparse

from mydatasets import process_dataset
from mydatasets import (
    BaseDataset,
    StyleCausalDataset,
    StyleAutoregressionDatasetMixin,
    StyleCausalDatasetMixin,
)

from mindnlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLM3Tokenizer,
    ChatGLM3ForConditionalGeneration,
)
from mindnlp.peft import get_peft_model, TaskType, IA3Config, LoraConfig
from mindnlp.peft import PeftModel, PeftConfig
from mindnlp.engine import Trainer, TrainingArguments, EvalPrediction
import evaluate

context.set_context(device_target="GPU")
# context.set_context(device_id=1)
ms.set_auto_parallel_context(
    parallel_mode=ms.ParallelMode.DATA_PARALLEL,
    gradients_mean=True,
    parameter_broadcast=True,
)
ms.common.set_seed(42)
init("nccl")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 4)
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B"
    )  # ZhipuAI/chatglm3-6b Qwen/Qwen2-7B-Instruct
    parser.add_argument("--peft_type", type=str, default="LoRA")
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")  # SEQ_2_SEQ_LM
    parser.add_argument(
        "--train_file", type=str, default="./data/sft/"
    )  # "./data/sft/XiYouJi/XiYouJi_Causal.json"
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def run(args):
    max_length = args.max_length
    batch_size = args.batch_size
    if args.train_file.endswith(".json"):
        train_path = args.train_file
    else:
        train_path = [
            os.path.join(args.train_file, p) for p in os.listdir(args.train_file)
        ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, mirror="modelscope", revision="master"
    )
    dataset = process_dataset(
        StyleCausalDatasetMixin(
            train_path, tokenizer, file_encoding="utf-8", max_length=max_length
        ),  # gb18030
        batch_size=batch_size,
        shuffle=True,
    )
    train_dataset, eval_dataset = dataset.split([0.9, 0.1])
    print(next(train_dataset.create_dict_iterator()))
    print(len(train_dataset))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        mirror="modelscope",
        revision="master",
        # ms_dtype=ms.float16,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.jit()

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataset_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type="polynomial",
        lr_scheduler_kwargs={
            "lr_end": args.learning_rate * 1e-5,
            "power": args.power,
        },
        # warmup_ratio=0.1,
        logging_steps=200,
        logging_dir=f"{args.save_dir}/runs",
        save_strategy="epoch",
        save_total_limit=1,
        # load_best_model_at_end=True,
        # bf16=True,
        # fp16=True,
        # fp16_opt_level="O1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()  # resume_from_checkpoint="./checkpoints/checkpoint-8880"

    peft_model_id = (
        f"{args.save_dir}/{args.model_name_or_path}_{args.peft_type}_{args.task_type}"
    )
    model.save_pretrained(peft_model_id)


if __name__ == "__main__":
    args = get_args()
    run(args)
