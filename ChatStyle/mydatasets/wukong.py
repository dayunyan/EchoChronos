import json
import mindspore as ms
from .base import BaseDataset


class WukongDataset(BaseDataset):
    def __init__(self, filepath, tokenizer, max_length, file_encoding="utf-8"):
        super(WukongDataset, self).__init__(
            filepath, tokenizer, max_length, file_encoding
        )
        self._load()

    def _load(self):
        with open(self.path, encoding=self.fencoding) as f:
            data = json.load(f)
            for d in data:
                self.instruction.append(d["cause"])
                self.output_sentence.append(d["effect"])

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()
