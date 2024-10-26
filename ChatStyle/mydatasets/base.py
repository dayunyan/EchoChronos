import json
import mindspore as ms


class BaseDataset:
    def __init__(self, filepath, tokenizer, max_length, file_encoding="utf-8"):
        self.path = filepath
        self.instruction = []
        self.input_sentence = []
        self.output_sentence = []
        self.fencoding = file_encoding
        self._load()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load(self):
        pass
        # with open(self.path, encoding=self.fencoding) as f:
        #     data = json.load(f)
        #     for d in data:
        #         self.instruction.append(d["question"])
        #         self.output_sentence.append(d["answer"])

    def __getitem__(self, index):
        inputs = self.instruction[index]
        output_sentence = self.output_sentence[index]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        labels = self.tokenizer(
            output_sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        labels = labels["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return (
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            labels,
            inputs,
            output_sentence,
        )

    def __len__(self):
        return len(self.instruction)
