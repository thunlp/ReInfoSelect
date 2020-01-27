from transformers import BertTokenizer

def bert_embloader(cfg)
    tokenizer = BertTokenizer.from_pretrained(cfg["word2vec"])
    return tokenizer
