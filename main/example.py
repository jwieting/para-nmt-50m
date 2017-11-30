import random
import unicodedata
import sys

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))

def lookup(words,w):
    w = w.lower()
    if w in words:
        return words[w]
    else:
        w = w.translate(tbl)
        if w in words:
            return words[w]
        return words['UUUNKKK']

def lookup_no_unk(words,w):
    w = w.lower()
    if w in words:
        return words[w]

class example(object):

    def __init__(self, phrase):
        self.phrase = phrase.strip().lower()
        self.embeddings = []
        self.representation = None

    def populate_embeddings(self, words, unk):
        phrase = self.phrase.lower()
        arr = phrase.split()
        for i in arr:
            if unk:
                self.embeddings.append(lookup(words,i))
            else:
                w = lookup_no_unk(words, i)
                if w:
                    self.embeddings.append(w)

    def populate_embeddings_ngrams(self, words, size, unk):
        phrase = " " + self.phrase.lower() + " "
        for j in range(len(phrase)):
            ngram = phrase[j:j+size]
            if len(ngram) != size:
                continue
            if unk:
                self.embeddings.append(lookup(words, ngram))
            else:
                w = lookup_no_unk(words, ngram)
                if w:
                    self.embeddings.append(w)
        if len(self.embeddings) == 0:
            self.embeddings = [words['UUUNKKK']]

    def unpopulate_embeddings(self):
        self.embeddings = []

    def populate_embeddings_scramble(self, words):
        phrase = self.phrase.lower()
        arr = phrase.split()
        random.shuffle(arr)
        for i in arr:
            self.embeddings.append(lookup(words,i))