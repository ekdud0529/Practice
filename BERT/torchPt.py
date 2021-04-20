from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    "beomi/kcbert-base",
    do_lower_case=False,
)

from transformers import BertConfig, BertModel
pretrained_model_config = BertConfig.from_pretrained(
    "beomi/kcbert-base"
)

import torch
model = BertModel.from_pretrained(
    "beomi/kcbert-base",
    config=pretrained_model_config,
)

sentences = ["안녕하세요", "하이!"]
features = tokenizer(
    sentences,
    max_length=10,
    padding="max_length",
    truncation=True,
)

features = {k: torch.tensor(v) for k, v in features.items()}

outputs = model(**features)

print(outputs[1])