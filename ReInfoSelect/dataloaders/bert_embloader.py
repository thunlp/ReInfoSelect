from transformers import BertTokenizer

def bert_embloader(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    return tokenizer
