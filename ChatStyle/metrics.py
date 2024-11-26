from typing import List
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import mindspore as ms


def compute_bleu_metrics(predictions: List[str], labels: List[str]):
    bleu_scores = []
    for prediction, reference in zip(predictions, labels):
        # 使用jieba进行中文分词
        prediction_tokens = jieba.cut(prediction)
        reference_tokens = jieba.cut(reference)

        # 将生成的列表转换为list
        prediction_tokens = [token for token in prediction_tokens]
        reference_tokens = [token for token in reference_tokens]

        # 计算BLEU分数，使用nltk的SmoothingFunction来处理零匹配问题
        score = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
        bleu_scores.append(score)

    return np.mean(bleu_scores)


if __name__ == "__main__":
    # 示例使用
    predictions = ["这是一个预测结果", "这是一个预测结果2"]
    labels = [
        ms.Tensor(
            np.asarray(
                [
                    "这是一个参考标签",
                ]
            ),
            dtype=ms.string,
        ),
        ms.Tensor(
            np.asarray(
                [
                    "这是一个参考标签2",
                ]
            ),
            dtype=ms.string,
        ),
    ]
    print("labels: ", labels)

    bleu = compute_bleu_metrics(predictions, labels)
    print(f"BLEU score: {bleu}")
