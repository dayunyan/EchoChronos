import mindspore as ms
import numpy as np
from mindspore.dataset import GeneratorDataset
from mindnlp.core.ops import tensor


from .base import BaseDataset
from .wukong import WukongDataset
from .style import (
    StyleSeq2SeqDataset,
    StyleCausalDataset,
    StyleAutoregressionDatasetMixin,
    StyleCausalDatasetMixin,
    StyleAlpacaSFTDataset,
)


def process_dataset(source, batch_size=32, shuffle=False):

    column_names = [
        "input_ids",
        "attention_mask",
        "labels",
        "text_inputs",
        "text_labels",
    ]

    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)

    # print(type(next(enumerate(dataset))[1][0]))
    # squeeze

    # dataset = dataset.map(
    #     operations=lambda x: (
    #         tokenizer(
    #             x,
    #             max_length=max_length,
    #             padding="max_length",
    #             truncation=True,
    #         )["input_ids"],
    #         tokenizer(
    #             x,
    #             max_length=max_length,
    #             padding="max_length",
    #             truncation=True,
    #         )["attention_mask"],
    #     ),
    #     input_columns="inputs",
    #     output_columns=["input_ids", "attention_mask"],
    # )
    # dataset = dataset.map(
    #     operations=lambda x: (
    #         tokenizer(
    #             x,
    #             max_length=max_length,
    #             padding="max_length",
    #             truncation=True,
    #         )["input_ids"],
    #     ),
    #     input_columns="output_sentence",
    #     output_columns="labels",
    # )
    # if batch_size is not None:
    #     dataset = dataset.batch(batch_size)
    return dataset
