from copy import deepcopy
import json
import os
import mindspore as ms
from .base import BaseDataset

DATASETS_MAP = {
    "XiYouJi": "西游记",
    "ShuiHuZhuan": "水浒传",
    "SanGuoYanYi": "三国演义",
    "HongLouMeng": "红楼梦",
}


class StyleAutoregressionDataset(BaseDataset):
    def __init__(self, filepath, tokenizer, file_encoding="utf-8", max_length=512):
        super().__init__(filepath, tokenizer, file_encoding, max_length)
        self._load()
        self.prompt_system = "<|im_start|>system\n{}<|im_end|>\n".format(
            "按《西游记》风格补写内容。"
        )
        self.prompt_assistant = "<|im_start|>{}<|im_end|><|endoftext|>"

    def _load(self):
        with open(self.path, "r", encoding=self.fencoding) as f:
            data = json.load(f)
            for d in data:
                if isinstance(d["output"], list):
                    for item in d["output"]:
                        self.output_sentence.append(item)
                else:
                    self.output_sentence.append(d["output"])

    def __getitem__(self, index):
        instruction = self.prompt_system
        inputs_sentence = instruction + self.prompt_assistant.format(
            self.output_sentence[index]
        )

        instruction_ids = self.tokenizer(instruction)["input_ids"]
        len_instruction_ids = len(instruction_ids)
        inputs = self.tokenizer(
            inputs_sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs["attention_mask"][:len_instruction_ids] = [0] * len_instruction_ids
        labels = deepcopy(inputs["input_ids"])

        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            labels,
            inputs_sentence,
            inputs_sentence,
        )

    def __len__(self):
        return len(self.output_sentence)


class StyleAutoregressionDatasetMixin(BaseDataset):
    def __init__(
        self, filepath: list[str], tokenizer, file_encoding="utf-8", max_length=512
    ):
        super().__init__(filepath, tokenizer, file_encoding, max_length)
        self._load()
        self.prompt_system = "<|im_start|>system\n{}<|im_end|>\n"
        self.prompt_assistant = "<|im_start|>{}<|im_end|><|endoftext|>"

    def _load(self):
        for path in self.path:
            bookname_en = os.path.basename(path)
            bookname_cn = DATASETS_MAP[os.path.basename(path)]
            with open(
                os.path.join(path, f"{bookname_en}.json"), "r", encoding=self.fencoding
            ) as f:
                data = json.load(f)
                for d in data:
                    self.instruction.append(
                        "按《{}》风格补写内容。".format(bookname_cn)
                    )
                    self.output_sentence.append(d["output"])

    def __getitem__(self, index):
        instruction = self.prompt_system.format(self.instruction[index])
        inputs_sentence = instruction + self.prompt_assistant.format(
            self.output_sentence[index]
        )

        instruction_ids = self.tokenizer(instruction)["input_ids"]
        len_instruction_ids = len(instruction_ids)
        inputs = self.tokenizer(
            inputs_sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs["attention_mask"][:len_instruction_ids] = [0] * len_instruction_ids
        labels = deepcopy(inputs["input_ids"])
        labels[:len_instruction_ids] = [-100] * len_instruction_ids

        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            labels,
            inputs_sentence,
            inputs_sentence,
        )

    def __len__(self):
        return len(self.output_sentence)


class StyleSeq2SeqDataset(BaseDataset):
    def __init__(
        self,
        filepath,
        tokenizer,
        instruction=None,
        file_encoding="utf-8",
        max_length=512,
    ):
        super().__init__(filepath, tokenizer, file_encoding, max_length)
        self.ins = instruction
        self.only_one_instruction = self.ins is not None
        self._load()
        self.prompt_system = "<|im_start|>system\n{}<|im_end|>\n"
        self.prompt_user = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        self.prompt_assistant = "assistant\n{}<|im_end|><|endoftext|>"

    def _load(self):
        with open(self.path, "r", encoding=self.fencoding) as f:
            data = json.load(f)
            for d in data:
                if isinstance(d["output"], list):
                    for i in range(len(d["output"])):
                        if self.only_one_instruction:
                            self.instruction.append(self.ins)
                        else:
                            self.instruction.append(d["instruction"])
                        self.input_sentence.append(d["input"])
                        self.output_sentence.append(d["output"][i])
                else:
                    if self.only_one_instruction:
                        self.instruction.append(self.ins)
                    else:
                        self.instruction.append(d["instruction"])
                    self.input_sentence.append(d["input"])
                    self.output_sentence.append(d["output"])

    def __getitem__(self, index):
        instruction = self.prompt_system.format(self.instruction[index])
        inputs_sentence = instruction + self.prompt_user.format(
            self.input_sentence[index]
        )
        output_sentence = self.prompt_assistant.format(self.output_sentence[index])

        inputs_ids = self.tokenizer(inputs_sentence)["input_ids"]
        output_ids = self.tokenizer(output_sentence)["input_ids"]
        len_inputs_ids = len(inputs_ids)
        len_output_ids = len(output_ids)
        print(type(inputs_ids), type(len_output_ids))
        inputs_ids += output_ids
        output_ids = [-100] * len_inputs_ids + output_ids
        attention_mask = [0] * len_inputs_ids + [1] * (len_output_ids)
        if (len_inputs_ids + len_output_ids) > self.max_length:
            inputs_ids = inputs_ids[: self.max_length]
            output_ids = output_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
        else:
            inputs_ids += [151643] * (self.max_length - len_inputs_ids - len_output_ids)
            output_ids += [151643] * (self.max_length - len_inputs_ids - len_output_ids)
            attention_mask += [0] * (self.max_length - len_inputs_ids - len_output_ids)

        return (
            inputs_ids,
            attention_mask,
            output_ids,
            inputs_sentence,
            output_sentence,
        )


class StyleCausalDataset(BaseDataset):
    def __init__(self, filepath, tokenizer, file_encoding="utf-8", max_length=512):
        super().__init__(filepath, tokenizer, file_encoding, max_length)
        self._load()
        self.prompt_system = "<|im_start|>system\n{}<|im_end|>\n".format(
            "假如你是《西游记》中的某个角色，请与我对话"
        )
        self.prompt_user = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        self.prompt_assistant = "assistant\n{}<|im_end|><|endoftext|>"

    def _load(self):
        with open(self.path, "r", encoding=self.fencoding) as f:
            data = json.load(f)
            for d in data:
                if isinstance(d, list):  # 多轮对话
                    inst_temp = []
                    input_temp = []
                    output_temp = []
                    for item in d:
                        inst_temp.append(item["instruction"])
                        input_temp.append(item["input"])
                        output_temp.append(item["output"])
                    self.instruction.append(inst_temp)
                    self.input_sentence.append(input_temp)
                    self.output_sentence.append(output_temp)
                else:
                    if isinstance(d["output"], list):
                        for item in d["output"]:
                            self.output_sentence.append(item)
                        self.instruction.append(d["instruction"])
                        self.input_sentence.append(d["input"])
                    else:
                        self.instruction.append(d["instruction"])
                        self.input_sentence.append(d["input"])
                        self.output_sentence.append(d["output"])

    def __getitem__(self, index):
        instruction = self.prompt_system
        if isinstance(self.input_sentence[index], list):
            for i in range(len(self.input_sentence[index]) - 1):
                instruction += self.prompt_user.format(self.input_sentence[index][i])
                instruction += self.prompt_assistant.format(
                    self.output_sentence[index][i]
                ).strip("<|endoftext|>")
            instruction += self.prompt_user.format(self.input_sentence[index][-1])
            output_sentence = self.output_sentence[index][-1]

            inputs_ids = self.tokenizer(inputs_sentence)["input_ids"]
            output_ids = self.tokenizer(output_sentence)["input_ids"]
            len_inputs_ids = len(inputs_ids)
            len_output_ids = len(output_ids)

            inputs_ids += output_ids
            output_ids = [-100] * len_inputs_ids + output_ids
            attention_mask = [0] * len_inputs_ids + [1] * (len_output_ids)
            if (len_inputs_ids + len_output_ids) > self.max_length:
                inputs_ids = inputs_ids[: self.max_length]
                output_ids = output_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                inputs_ids += [151643] * (
                    self.max_length - len_inputs_ids - len_output_ids
                )
                output_ids += [151643] * (
                    self.max_length - len_inputs_ids - len_output_ids
                )
                attention_mask += [0] * (
                    self.max_length - len_inputs_ids - len_output_ids
                )
        else:
            inputs_sentence = (
                instruction
                + self.prompt_user.format(self.input_sentence[index])
                + self.prompt_assistant.format(self.output_sentence[index])
            )

            instruction_ids = self.tokenizer(instruction)["input_ids"]
            len_instruction_ids = len(instruction_ids)
            inputs = self.tokenizer(
                inputs_sentence,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            inputs["attention_mask"][:len_instruction_ids] = [0] * len_instruction_ids
            labels = deepcopy(inputs["input_ids"])

        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            labels,
            inputs_sentence,
            inputs_sentence,
        )


class StyleCausalDatasetMixin(BaseDataset):
    def __init__(
        self, filepath: list[str], tokenizer, file_encoding="utf-8", max_length=512
    ):
        super().__init__(filepath, tokenizer, file_encoding, max_length)
        self._load()
        self.prompt_system = "<|im_start|>system\n{}<|im_end|>\n"
        self.prompt_user = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        self.prompt_assistant = "assistant\n{}<|im_end|><|endoftext|>"

    def _load(self):
        for path in self.path:
            bookname_en = os.path.basename(path)
            bookname_cn = DATASETS_MAP[os.path.basename(path)]
            with open(
                os.path.join(path, f"{bookname_en}_Conversation.json"),
                "r",
                encoding=self.fencoding,
            ) as f:
                data = json.load(f)
                for d in data:
                    if isinstance(d, list):  # 多轮对话
                        inst_temp = []
                        input_temp = []
                        output_temp = []
                        for item in d:
                            inst_temp.append(
                                "假如你是《{}》中的某个角色，请与我对话".format(
                                    bookname_cn
                                )
                            )
                            input_temp.append(item["input"])
                            output_temp.append(item["output"])
                        self.instruction.append(inst_temp)
                        self.input_sentence.append(input_temp)
                        self.output_sentence.append(output_temp)
                    else:
                        self.instruction.append(
                            [
                                "假如你是《{}》中的某个角色，请与我对话".format(
                                    bookname_cn
                                )
                            ]
                        )
                        self.input_sentence.append([d["input"]])
                        self.output_sentence.append([d["output"]])

    def __getitem__(self, index):
        instruction = self.prompt_system.format(self.instruction[index][0])
        for i in range(len(self.input_sentence[index]) - 1):
            instruction += self.prompt_user.format(self.input_sentence[index][i])
            instruction += self.prompt_assistant.format(self.output_sentence[index][i])[
                :-13
            ]  # 去掉中间的<|endoftext|>
        instruction += self.prompt_user.format(self.input_sentence[index][-1])
        output_sentence = self.output_sentence[index][-1]

        inputs_ids = self.tokenizer(instruction)["input_ids"]
        output_ids = self.tokenizer(output_sentence)["input_ids"]
        len_inputs_ids = len(inputs_ids)
        len_output_ids = len(output_ids)

        inputs_ids += output_ids
        output_ids = [-100] * len_inputs_ids + output_ids
        attention_mask = [0] * len_inputs_ids + [1] * (len_output_ids)
        if (len_inputs_ids + len_output_ids) > self.max_length:
            inputs_ids = inputs_ids[: self.max_length]
            output_ids = output_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
        else:
            inputs_ids += [151643] * (self.max_length - len_inputs_ids - len_output_ids)
            output_ids += [151643] * (self.max_length - len_inputs_ids - len_output_ids)
            attention_mask += [0] * (self.max_length - len_inputs_ids - len_output_ids)

        return (
            inputs_ids,
            attention_mask,
            output_ids,
            instruction,
            output_sentence,
        )
