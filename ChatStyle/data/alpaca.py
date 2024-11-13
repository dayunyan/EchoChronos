import os
import json

DATASETS_MAP = {
    "XiYouJi": "西游记",
    "ShuiHuZhuan": "水浒传",
    "SanGuoYanYi": "三国演义",
    "HongLouMeng": "红楼梦",
}


# 将conversation数据合并为一个alpaca格式的指令监督微调数据集
def alpaca_sft_merge(root, save_path):
    alpaca_data = []
    for dir in os.listdir(root):
        if dir in DATASETS_MAP:
            with open(
                os.path.join(root, dir, f"{dir}_Conversation.json"),
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
                instruction = "假如你是<{}>中的某个角色,请与我对话。".format(
                    DATASETS_MAP[dir]
                )
                for item in data:
                    history = []
                    for i in range(len(item) - 1):
                        history.append(
                            [
                                (
                                    f"{instruction}\n{item[i]['input']}"
                                    if i == 0
                                    else item[i]["input"]
                                ),
                                item[i]["output"],
                            ]
                        )
                    alpaca_data.append(
                        {
                            "instruction": instruction if len(history) == 0 else "",
                            "input": item[-1]["input"],
                            "output": item[-1]["output"],
                            "history": history,
                        }
                    )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=4)
        f.close()


if __name__ == "__main__":
    alpaca_sft_merge(".", "alpaca_sft_data.json")
