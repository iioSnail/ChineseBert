import os.path

from modeling_glycebert import GlyceBertForMaskedLM
from bert_tokenizer import ChineseBertTokenizer

source_path = "./ChineseBERT-large"
bert_path = "./iioSnail/ChineseBERT-large"
os.makedirs(bert_path, exist_ok=True)


def load_tokenizer():
    tokenizer = ChineseBertTokenizer.from_pretrained(source_path)
    return tokenizer


def load_model():
    model = GlyceBertForMaskedLM.from_pretrained(source_path)
    return model


def test_model(tokenizer, model):
    sentence = '我喜 [MASK] 猫'
    input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)
    length = input_ids.shape[0]
    input_ids = input_ids.view(1, length)
    pinyin_ids = pinyin_ids.view(1, length, 8)
    output_hidden = model(input_ids, pinyin_ids).logits
    print(tokenizer.convert_ids_to_tokens(output_hidden.argmax(-1)[0, 1:-1]))
    print(output_hidden.size())
    print("-" * 30)


def export_tokenizer(tokenizer):
    tokenizer.register_for_auto_class("AutoTokenizer")
    tokenizer.save_vocabulary(bert_path)
    tokenizer.save_pretrained(bert_path)


def export_model(model):
    model.register_for_auto_class("AutoModel")
    model.save_pretrained(bert_path)


def main():
    # In the first, you should put the model file provided by Author into {source_path}
    tokenizer = load_tokenizer()
    model = load_model()
    test_model(tokenizer, model)
    export_tokenizer(tokenizer)
    export_model(model)
    print("Export success!")
    # In final, you should copy "config" directory to the bert_path directory.


if __name__ == '__main__':
    main()
