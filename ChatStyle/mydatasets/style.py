from copy import deepcopy
import json
import mindspore as ms
from .base import BaseDataset


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
            "根据上文，生成一段风格相似的下文。"
        )
        self.prompt_user = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        self.prompt_assistant = "assistant\n{}<|im_end|><|endoftext|>"

    def _load(self):
        with open(self.path, "r", encoding=self.fencoding) as f:
            data = json.load(f)
            for d in data:
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
