import re
from nltk.corpus import stopwords
from krovetzstemmer import Stemmer

class Tokenizer:
    def __init__(self, vocab_path, unk="<UNK>", pad="<PAD>"):
        self.vocab_path = vocab_path
        self.unk = unk
        self.pad = pad
        self.word2idx = self.load_vocab(vocab_path)
        self.sws = {}
        for w in stopwords.words('english'):
            self.sws[w] = 1
        self.stemmer = Stemmer()

    def load_vocab(self, vocab_path):
        word2idx = {}
        word2idx[self.pad] = 0
        word2idx[self.unk] = 1
        with open(vocab_path) as fin:
            for step, line in enumerate(fin):
                tokens = line.strip().split()
                word2idx[tokens[0]] = step + 2
        return word2idx

    def tok2idx(self, toks, word2idx):
        input_ids = []
        for tok in toks:
            if tok in word2idx:
                input_ids.append(word2idx[tok])
            else:
                input_ids.append(word2idx['<UNK>'])
        return input_ids

    def tokenize(self, line):
        regex_drop_char = re.compile('[^a-z0-9\s]+')
        regex_multi_space = re.compile('\s+')
        toks = regex_multi_space.sub(' ', regex_drop_char.sub(' ', line.lower())).strip().split()
        wordsFiltered = []
        for w in toks:
            if w not in self.sws:
                w = self.stemmer.stem(w)
                wordsFiltered.append(w)
        return wordsFiltered

    def convert_tokens_to_ids(self, toks):
        input_ids = []
        for tok in toks:
            if tok in self.word2idx:
                input_ids.append(self.word2idx[tok])
            else:
                input_ids.append(self.word2idx[self.unk])
        return input_ids
