# 以”为分隔符划分每一段。

import json
from utils.filetools import open_file, save_json
from tqdm import tqdm

KEYS = ["石猴", "猴王", "悟空", "大圣", "那猴", "行者"]
if __name__ == "__main__":
    file_path = "./西游记.txt"
    text_ls = open_file(file_path)
    print("文本行数：", len(text_ls))
    sentences = []
    temp = ""
    for line in tqdm(text_ls):
        line = line.strip("\n")
        if "“" in line and "”" in line:
            line = line.split("”")
            for i in range(len(line) - 1):
                sentences.append(line[i] + "”")
            if line[-1] != "":
                sentences.append(line[-1])
        elif "“" in line and "”" not in line:
            temp = line
        elif "“" not in line and "”" in line:
            if temp != "":
                sentences.append(temp + line)
                temp = ""
            else:
                sentences.append("“" + line)
        else:
            if temp != "":
                temp += line
            else:
                sentences.append(line)
    print("句子数量：", len(sentences))

    data = []
    for id, sentence in enumerate(sentences):
        if sentence[0] in ("，", "。", "！", "？", "：", "；"):
            sentence = sentence[1:]
            sentences[id] = sentence
        if any(key in sentence for key in KEYS):
            data.append(
                {
                    "cau_id": id - 1,
                    "cause": sentences[id - 1],
                    "eff_id": id,
                    "effect": sentence,
                }
            )
    print("数据数量：", len(data))

    save_json(data, "./XiYouJi/SunWuKong.json")
