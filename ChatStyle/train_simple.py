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
from mydatasets import BaseDataset, StyleCausalDataset
from losser import CrossEntropyLossForCausalLM

from mindnlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLM3Tokenizer,
    ChatGLM3ForConditionalGeneration,
    Qwen2Config,
)
from mindnlp.peft import get_peft_model, TaskType, IA3Config, LoraConfig
from mindnlp.peft import PeftModel, PeftConfig
from mindnlp.engine import Trainer, TrainingArguments, EvalPrediction
from mindnlp import amp
import evaluate

from metrics import compute_bleu_metrics

context.set_context(
    device_target="GPU"
)  # context.set_context(mode=ms.GRAPH_MODE) 新版本mindspore逐步禁用
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
        "--train_file", type=str, default="./XiYouJi/XiYouji_Preference.json"
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def run(args):
    # profiler = ms.Profiler(output_path="./profiler", start_profile=False)
    max_length = args.max_length
    batch_size = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, mirror="modelscope", revision="master"
    )
    dataset = process_dataset(
        StyleCausalDataset(
            args.train_file, tokenizer, file_encoding="utf-8", max_length=max_length
        ),  # gb18030
        batch_size=batch_size,
        shuffle=True,
    )
    train_dataset, eval_dataset = dataset.split([0.9, 0.1])
    print(next(train_dataset.create_dict_iterator()))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        mirror="modelscope",
        revision="master",  # config=config
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

    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    power = args.power
    num_training_steps = num_epochs * len(train_dataset)

    # lr_scheduler = PolynomialDecayLR(
    #     learning_rate, learning_rate * 0.001, num_training_steps, power
    # )
    optimizer = optim.AdamW(model.trainable_params(), lr=learning_rate)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps * 0.1,
        num_training_steps=num_training_steps,
        lr_end=learning_rate * 1e-4,
        power=power,
    )

    num_batches = len(train_dataset)
    num_batches_eval = len(eval_dataset)

    def compute_ce_loss(logits, labels):
        """deprecated test method

        Args:
            logits (Tensor)
            labels (Tensor)

        Returns:
            Tensor: The cross entropy loss
        """
        loss_fct = CrossEntropyLossForCausalLM()
        # Enable model parallelism
        loss = loss_fct(logits, labels)

        return loss

    def forward_fn(input_ids, attention_mask, labels):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # loss = compute_ce_loss(output.logits, labels)
        return output.loss, output.logits

    grad_fn = ms.value_and_grad(
        forward_fn, None, model.trainable_params(), has_aux=True
    )

    def train_step(input_ids, attention_mask, labels):
        (loss, logits), grads = grad_fn(input_ids, attention_mask, labels)
        optimizer.step(grads)
        return loss, logits

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        model.set_train(True)
        total_loss, total_step = 0, 0
        with tqdm(total=num_batches, leave=False, position=1, desc="train_step") as t:
            for step, pack in enumerate(train_dataset.create_dict_iterator()):
                input_ids = pack["input_ids"]
                attention_mask = pack["attention_mask"]
                labels = pack["labels"]
                loss, logits = train_step(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                total_loss += loss.asnumpy()
                lr_scheduler.step()
                total_step += 1
                curr_loss = total_loss / total_step
                t.set_postfix({"train-loss": f"{curr_loss:.2f}"})
                t.update(1)
                # if profiler is not None:
                #     if step == 10:
                #         profiler.start()
                #     if step == 100:
                #         profiler.stop()

        model.set_train(False)
        eval_loss = 0
        total_step = 0
        eval_preds = []
        total_text_labels = []
        with tqdm(
            total=num_batches_eval, leave=False, position=1, desc="eval_step"
        ) as t:
            for step, pack in enumerate(eval_dataset.create_dict_iterator()):
                input_ids = pack["input_ids"]
                attention_mask = pack["attention_mask"]
                labels = pack["labels"]
                text_inputs = pack["text_inputs"]
                text_labels = pack["text_labels"]
                with ms._no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                loss = compute_ce_loss(outputs.logits, labels)
                eval_loss += loss.asnumpy()
                total_step += 1
                curr_eval_loss = eval_loss / total_step
                eval_preds.extend(
                    tokenizer.batch_decode(
                        outputs.logits.argmax(axis=-1).asnumpy(),
                        skip_special_tokens=True,
                    )
                )

                total_text_labels.extend(text_labels.tolist())
                t.set_postfix({"eval-loss": f"{curr_eval_loss:.2f}"})
                t.update(1)
        bleu_avg = compute_bleu_metrics(eval_preds, total_text_labels)
        # accuracy = correct / total * 100
        # print(f"{accuracy=} % on the evaluation dataset")
        eval_epoch_loss = eval_loss / eval_dataset.get_dataset_size()
        eval_ppl = np.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / train_dataset.get_dataset_size()
        train_ppl = np.exp(train_epoch_loss)
        tqdm.write(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=} {bleu_avg=}"
        )

    peft_model_id = f"{args.model_name_or_path}_{args.peft_type}_{args.task_type}"
    model.save_pretrained(peft_model_id)
    # profiler.analyse()


if __name__ == "__main__":
    args = get_args()
    run(args)
