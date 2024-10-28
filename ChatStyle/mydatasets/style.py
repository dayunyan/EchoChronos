import json
import mindspore as ms
from .base import BaseDataset


class StyleTransferTaskDataset(BaseDataset):
    def __init__(self, filepath, instruction, file_encoding="utf-8"):
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
