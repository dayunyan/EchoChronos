import mindspore as ms
import numpy as np
from mindspore.dataset import GeneratorDataset

from .base import BaseDataset
from .wukong import WukongDataset


def process_dataset(source, batch_size=32, shuffle=False):

    column_names = [
        "input_ids",
        "attention_mask",
        "labels",
        "text_inputs",
        "text_labels",
    ]

    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)

    # dataset = dataset.batch(batch_size)
    # print(type(next(enumerate(dataset))[1][0]))
    # squeeze
    for name in column_names[0:-2]:
        dataset = dataset.map(
            operations=lambda x: np.squeeze(x, 0),
            input_columns=name,
            output_columns=name,
        )

    return dataset
