from mindnlp.transformers import AutoTokenizer
from mydatasets import StyleSeq2SeqDataset, StyleCausalDataset

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B", mirror="modelscope", revision="master"
)

# dataset1 = StyleSeq2SeqDataset(
#     "./data/sft/XiYouJi/XiYouji_Preference.json",
#     tokenizer,
#     instruction="将白话文转换成文言文。",
#     max_length=256,
# )
# print(dataset1[0])
# dataset2 = StyleCausalDataset(
#     "./data/sft/XiYouJi/XiYouJi_Causal.json", tokenizer, max_length=256
# )
# print(dataset2[0])

print(tokenizer)
prompt = "你好，请介绍一下自己。"
prompt2 = "帮我制定一份旅行计划。"
# message = [
#     [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
#     [{"role": "user", "content": prompt2}],
# ]
message = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    truncation=True,
    return_tensors="ms",
    add_generation_prompt=True,
)
print(f"template text: {text}")
text = tokenizer(text, padding=True, truncation=True, return_tensors="ms")
print(f"tokenized text: {text}")
# print(text["input_ids"].squeeze(0))
text_dec = tokenizer.batch_decode(
    text["input_ids"].asnumpy(), skip_special_tokens=False
)
print(text_dec)
