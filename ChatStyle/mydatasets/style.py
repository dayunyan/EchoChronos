import json
import mindspore as ms
from .base import BaseDataset


class StyleSeq2SeqDataset(BaseDataset):
    def __init__(self, filepath, instruction=None, file_encoding="utf-8"):
        super().__init__(filepath, file_encoding)
        self.ins = instruction
        self.only_one_instruction = self.ins is not None
        self._load()

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
        instruction = self.instruction[index]
        inputs = (
            self.input_sentence[index]
            if instruction == ""
            else instruction + "\n" + self.input_sentence[index]
        )
        output_sentence = self.output_sentence[index]

        return (
            inputs,
            output_sentence,
            inputs,
            output_sentence,
        )


class StyleTransferDataset(StyleSeq2SeqDataset):
    def __init__(self, filepath, instruction=None, file_encoding="utf-8"):
        super().__init__(filepath, instruction, file_encoding)

    def __getitem__(self, index):
        instruction = self.instruction[index]
        inputs = (
            self.input_sentence[index]
            if instruction == ""
            else "【System】" + instruction + "【User】" + self.input_sentence[index]
        )
        output_sentence = inputs + "【Assitant】" + self.output_sentence[index]

        return (
            inputs,
            output_sentence,
            inputs,
            output_sentence,
        )


class StyleCausalDataset(BaseDataset):
    def __init__(self, filepath, file_encoding="utf-8"):
        super().__init__(filepath, file_encoding)
        self._load()

    def _load(self):
        with open(self.path, "r", encoding=self.fencoding) as f:
            data = json.load(f)
            for d in data:
                if isinstance(d["output"], list):
                    for item in d["output"]:
                        self.input_sentence.append(item)
                        self.output_sentence.append(item)
                else:
                    self.input_sentence.append(d["output"])
                    self.output_sentence.append(d["output"])

    def __getitem__(self, index):
        inputs = self.input_sentence[index]
        output_sentence = self.output_sentence[index]

        return (
            inputs,
            output_sentence,
            inputs,
            output_sentence,
        )
