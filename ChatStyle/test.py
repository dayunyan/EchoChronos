from mindnlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-7B-Instruct", mirror="modelscope", revision="master"
)
print(tokenizer)
message = ["Das kleine Kind klettert an roten Seilen auf einem Spielplatz.", "你是谁？"]
text = tokenizer(message, padding=True, truncation=True, return_tensors="ms")
print(text)
print(text["input_ids"].squeeze(0))
