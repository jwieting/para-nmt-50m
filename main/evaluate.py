import numpy as np
import io
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from example import example

def get_seqs(p1, p2, words, params):
    p1 = example(p1)
    p2 = example(p2)

    if params.wordtype == "words":
        p1.populate_embeddings(words, True)
        p2.populate_embeddings(words, True)
    else:
        p1.populate_embeddings_ngrams(words, 3, True)
        p2.populate_embeddings_ngrams(words, 3, True)

    return p1.embeddings, p2.embeddings

def get_correlation(model, words, f, params):
    f = io.open(f, 'r', encoding='utf-8')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = get_seqs(p1, p2, words, params)
        seq1.append(X1)
        seq2.append(X2)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = model.prepare_data(seq1)
            x2,m2 = model.prepare_data(seq2)
            scores = model.scoring_function(x1,x2,m1,m2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = model.prepare_data(seq1)
        x2,m2 = model.prepare_data(seq2)
        scores = model.scoring_function(x1,x2,m1,m2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def evaluate_all(model, words, params):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["annotated-ppdb-dev",
            "annotated-ppdb-test"]

    for i in farr:
        p,s = get_correlation(model, words, prefix + i, params)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j,k in zip(parr,sarr,farr):
        s += str(i)+" "+str(j)+" "+k+" | "

    print s
    return parr[0]