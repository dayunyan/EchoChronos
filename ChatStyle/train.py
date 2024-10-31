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
from mydatasets import BaseDataset, StyleTransferTaskDataset

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

from nltk.translate.bleu_score import corpus_bleu

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
context.set_context(device_id=1)
# ms.set_auto_parallel_context(
#     parallel_mode=ms.ParallelMode.AUTO_PARALLEL, gradients_mean=True
# )
# init("nccl")


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
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def run(args):
    max_length = args.max_length
    batch_size = args.batch_size
    # dataset = ds.load_dataset("bentrevett/multi30k")
    # train_dataset = dataset["train"].shuffle(32)
    # eval_dataset = dataset["validation"].shuffle(32)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, mirror="modelscope", revision="master"
    )
    dataset = process_dataset(
        StyleTransferTaskDataset(
            args.train_file, instruction="将白话文转换成文言文。", file_encoding="utf-8"
        ),  # gb18030
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    train_dataset, eval_dataset = dataset.split([0.9, 0.1])
    print(next(train_dataset.create_dict_iterator()))
    # print(next(eval_dataset.create_dict_iterator()))

    # instruction = ["[system]: 将下面的英文句子翻译成德文。[user]: ", "[assistant]: "]
    # train_dataset = train_dataset.map(
    #     lambda x: (
    #         tokenizer(
    #             str(x).join(instruction),
    #             max_length=max_length,
    #             padding="max_length",
    #             truncation=True,
    #         )["input_ids"],
    #         tokenizer(
    #             str(x).join(instruction),
    #             max_length=max_length,
    #             padding="max_length",
    #             truncation=True,
    #         )["attention_mask"],
    #     ),
    #     input_columns="en",
    #     output_columns=["input_ids", "attention_mask"],
    # )
    # train_dataset = train_dataset.map(
    #     lambda x: tokenizer(
    #         str(x).join(instruction),
    #         max_length=max_length,
    #         padding="max_length",
    #         truncation=True,
    #     )["input_ids"],
    #     input_columns="de",
    #     output_columns="labels",
    # )
    # eval_dataset = eval_dataset.map(
    #     lambda x: (
    #         tokenizer(
    #             str(x).join(instruction),
    #             max_length=max_length,
    #             padding="max_length",
    #             truncation=True,
    #         )["input_ids"],
    #         tokenizer(
    #             str(x).join(instruction),
    #             max_length=max_length,
    #             padding="max_length",
    #             truncation=True,
    #         )["attention_mask"],
    #     ),
    #     input_columns="en",
    #     output_columns=["input_ids", "attention_mask"],
    # )
    # eval_dataset = eval_dataset.map(
    #     lambda x: tokenizer(
    #         str(x).join(instruction),
    #         max_length=max_length,
    #         padding="max_length",
    #         truncation=True,
    #     )["input_ids"],
    #     input_columns="de",
    #     output_columns="labels",
    # )
    # print(next(train_dataset.create_dict_iterator()))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, mirror="modelscope", revision="master"
    )

    # peft_config = IA3Config(
    #     peft_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     target_modules=["q_proj", "v_proj"],  # ["query_key_value"]
    #     feedforward_cells=[],
    # )
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

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        dataset_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type="polynomial",
        lr_scheduler_kwargs={
            "lr_end": args.learning_rate * 1e-5,
            "power": args.power,
        },
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=1,
        # load_best_model_at_end=True,
        fp16=True,
        fp16_opt_level="O3",
    )

    metric = evaluate.load("accuracy")

    def compute_bleu_metrics(eval_pred: EvalPrediction):
        predictions, labels = eval_pred
        bleu_scores = []
        for prediction, label in zip(predictions, labels):
            score = corpus_bleu([label], prediction)
            bleu_scores.append(score)
        return np.mean(bleu_scores)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_bleu_metrics,
    )
    trainer.train()  # resume_from_checkpoint="./checkpoints/checkpoint-8880"

    peft_model_id = (
        f"{args.save_dir}/{args.model_name_or_path}_{args.peft_type}_{args.task_type}"
    )
    model.save_pretrained(peft_model_id)

    # num_epochs = args.num_epochs
    # learning_rate = args.learning_rate
    # power = args.power
    # num_training_steps = num_epochs * len(train_dataset)

    # # lr_scheduler = PolynomialDecayLR(
    # #     learning_rate, learning_rate * 0.001, num_training_steps, power
    # # )
    # optimizer = optim.AdamW(model.trainable_params(), lr=learning_rate)
    # lr_scheduler = get_polynomial_decay_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=10,
    #     num_training_steps=num_training_steps,
    #     lr_end=learning_rate * 0.001,
    #     power=power,
    # )

    # num_batches = len(train_dataset)
    # num_batches_eval = len(eval_dataset)

    # def forward_fn(input_ids, attention_mask, labels):
    #     output = model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #     )
    #     return output.loss, output.logits

    # grad_fn = ms.value_and_grad(forward_fn, None, optimizer.param_groups, has_aux=True)

    # for epoch in range(num_epochs):
    #     model.set_train(True)
    #     total_loss, total_step = 0, 0
    #     correct = 0
    #     total = 0
    #     with tqdm(total=num_batches) as t:
    #         for step, (input_ids, attention_mask, labels, _) in enumerate(
    #             train_dataset
    #         ):
    #             input_ids = input_ids.squeeze(axis=1)
    #             labels = labels.squeeze(axis=1)
    #             attention_mask = attention_mask.squeeze(axis=1)
    #             (loss, logits), grad = grad_fn(input_ids, attention_mask, labels)
    #             optimizer(grad)
    #             total_loss += loss.asnumpy()
    #             total_step += 1
    #             curr_loss = total_loss / total_step
    #             t.set_postfix({"train-loss": f"{curr_loss:.2f}"})
    #             t.update(1)
    #     model.set_train(False)
    #     eval_loss = 0
    #     total_step = 0
    #     eval_preds = []
    #     text_labels = []
    #     with tqdm(total=num_batches_eval) as t:
    #         for step, (input_ids, attention_mask, labels, text) in enumerate(
    #             eval_dataset
    #         ):
    #             input_ids = input_ids.squeeze(axis=1)
    #             labels = labels.squeeze(axis=1)
    #             attention_mask = attention_mask.squeeze(axis=1)
    #             outputs = model(
    #                 input_ids=input_ids, attention_mask=attention_mask, labels=labels
    #             )
    #             loss = outputs.loss
    #             eval_loss += loss.asnumpy()
    #             total_step += 1
    #             eval_loss = total_loss / total_step
    #             eval_preds.extend(
    #                 tokenizer.batch_decode(
    #                     np.argmax(outputs.logits.asnumpy(), -1),
    #                     skip_special_tokens=True,
    #                 )
    #             )
    #             text_str = str(text.asnumpy())
    #             text_str = (
    #                 text_str.replace("[", "")
    #                 .replace("]", "")
    #                 .replace(",", "")
    #                 .replace("'", "")
    #             )
    #             labels = text_str.split(" ")
    #             text_labels.extend(labels)
    #             t.set_postfix({"eval-loss": f"{eval_loss:.2f}"})
    #             t.update(1)
    #     for pred, text_label in zip(eval_preds, text_labels):
    #         if pred.strip() == text_label.strip():
    #             correct += 1
    #         total += 1
    #     # accuracy = correct / total * 100
    #     # print(f"{accuracy=} % on the evaluation dataset")
    #     eval_epoch_loss = eval_loss / eval_dataset.get_dataset_size()
    #     eval_ppl = np.exp(eval_epoch_loss)
    #     train_epoch_loss = total_loss / train_dataset.get_dataset_size()
    #     train_ppl = np.exp(train_epoch_loss)
    #     print(
    #         f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
    #     )

    # peft_model_id = f"{args.model_name_or_path}_{args.peft_type}_{args.task_type}"
    # model.save_pretrained(peft_model_id)


def eval(args):
    import datasets

    peft_model_id = f"{args.model_name_or_path}_{args.peft_type}_{args.task_type}"
    tokenizer = ChatGLM3Tokenizer.from_pretrained(
        args.model_name_or_path, mirror="modelscope", revision="master"
    )
    model = ChatGLM3ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(model, peft_model_id)

    text_column = "input_ids"
    model.set_train(False)
    i = 13

    def load_dataset(filepath):
        inputs = []
        labels = []
        with open(filepath, encoding="gb18030") as f:
            data = json.load(f)
            for d in data:
                inputs.append(d["question"])
                labels.append(d["answer"])

        dataset = datasets.Dataset.from_dict(
            {
                "inputs": inputs,
                "label": labels,
            }
        )
        return dataset

    dataset = load_dataset("./XiYouJi/XiYouJi.json")
    dataset = dataset.train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="ms")
    print(dataset["validation"][text_column][i])
    print(inputs)

    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=128)
    # outputs,_ = model.chat(tokenizer,query=dataset["validation"][text_column][i], max_length=128)
    print(outputs)
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(text_output)


if __name__ == "__main__":
    args = get_args()
    run(args)
    # eval(args)
