import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class CrossEntropyLossForCausalLM(nn.Cell):
    def __init__(self, smoothing_factor=0.1, ignore_index=-100, reduction="mean"):
        super(CrossEntropyLossForCausalLM, self).__init__()
        self.log_softmax = ops.LogSoftmax(axis=-1)
        self.reduce_sum = ops.ReduceSum
        self.reduce_mean = ops.ReduceMean
        self.gather = ops.gather  # Used to gather the log probabilities
        self.one = ops.ones_like
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.smoothing_factor = smoothing_factor
        self.ignore_index = ignore_index
        self.reduction = reduction

    def construct(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        # Flatten the tokens
        batch_size, seq_len, vocab_size = shift_logits.shape
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Calculate log softmax
        log_probs = -self.log_softmax(shift_logits)

        # Create a mask for ignore_index
        mask = shift_labels.eq(self.ignore_index)

        shift_labels = ops.clamp(shift_labels, min=0)
        nll_loss = self.gather(log_probs, shift_labels, axis=1)
        smoothed_loss = self.reduce_sum(keep_dims=True)(log_probs, axis=1)

        nll_loss = nll_loss.masked_fill(mask, 0.0)
        smoothed_loss = smoothed_loss.masked_fill(mask, 0.0)

        num_active_elements = mask.numel() - mask.long().sum()
        nll_loss = self.reduce_sum()(nll_loss) / num_active_elements
        smoothed_loss = self.reduce_sum()(smoothed_loss) / (
            num_active_elements * vocab_size
        )
        if self.reduction == "mean":
            nll_loss /= num_active_elements
            smoothed_loss /= num_active_elements * vocab_size

        return (
            1 - self.smoothing_factor
        ) * nll_loss + self.smoothing_factor * smoothed_loss


# 使用示例
if __name__ == "__main__":
    # 假设您的模型输出的logits和真实的labels如下
    logits = Tensor(
        np.random.rand(32, 10, 1000), dtype=ms.float32
    )  # batch_size, sequence_length, vocab_size
    labels = Tensor(
        np.random.randint(0, 1000, size=(32, 10)), dtype=ms.int64
    )  # batch_size, sequence_length

    # 实例化损失函数
    ce_loss = CrossEntropyLossForCausalLM()

    # 计算损失
    loss = ce_loss(logits, labels)
    print(loss)
