import json
import mindspore as ms


class BaseDataset:
    def __init__(self, filepath, file_encoding="utf-8"):
        self.path = filepath
        self.instruction = []
        self.input_sentence = []
        self.output_sentence = []
        self.fencoding = file_encoding

    def _load(self):
        """
        with open(self.path, encoding=self.fencoding) as f:
            data = json.load(f)
            for d in data:
                self.input_sentence.append(d["question"])
                self.output_sentence.append(d["answer"])
        """
        pass

    def __getitem__(self, index):
        instruction = self.instruction[index]
        inputs = self.input_sentence[index]
        inputs = "[System]: " + instruction + "[User]: " + inputs + "[Assistant]: "
        output_sentence = self.output_sentence[index]

        return (
            inputs,
            output_sentence,
            inputs,
            output_sentence,
        )
        # model_inputs = self.tokenizer(
        #     inputs,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        #     # return_tensors="np",
        # )
        # labels = self.tokenizer(
        #     output_sentence,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        #     # return_tensors="np",
        # )
        # labels = labels["input_ids"]
        # labels[labels == self.tokenizer.pad_token_id] = -100

        # return (
        #     model_inputs["input_ids"],
        #     model_inputs["attention_mask"],
        #     labels,
        #     inputs,
        #     output_sentence,
        # )

    def __len__(self):
        return len(self.instruction)
