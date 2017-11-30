import io
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from example import example

def get_seqs(p1, p2, ngram_words, word_words, params):

    if params.combination_type == "ngram-word":

        np1 = example(p1)
        np2 = example(p2)

        wp1 = example(p1)
        wp2 = example(p2)

        np1.populate_embeddings_ngrams(ngram_words, 3, True)
        np2.populate_embeddings_ngrams(ngram_words, 3, True)
        wp1.populate_embeddings(word_words, True)
        wp2.populate_embeddings(word_words, True)

        return np1.embeddings, wp1.embeddings, np2.embeddings, wp2.embeddings

    elif params.combination_type == "ngram-lstm":

        np1 = example(p1)
        np2 = example(p2)

        wp1 = example(p1)
        wp2 = example(p2)

        np1.populate_embeddings_ngrams(ngram_words, 3, True)
        np2.populate_embeddings_ngrams(ngram_words, 3, True)
        wp1.populate_embeddings(word_words, True)
        wp2.populate_embeddings(word_words, True)

        return np1.embeddings, wp1.embeddings, np2.embeddings, wp2.embeddings

    elif params.combination_type == "word-lstm":

        np1 = example(p1)
        np2 = example(p2)

        wp1 = example(p1)
        wp2 = example(p2)

        np1.populate_embeddings(word_words, True)
        np2.populate_embeddings(word_words, True)
        wp1.populate_embeddings(word_words, True)
        wp2.populate_embeddings(word_words, True)

        return np1.embeddings, wp1.embeddings, np2.embeddings, wp2.embeddings

    elif params.combination_type == "ngram-word-lstm":

        np1 = example(p1)
        np2 = example(p2)

        wp1 = example(p1)
        wp2 = example(p2)

        lp1 = example(p1)
        lp2 = example(p2)

        np1.populate_embeddings_ngrams(ngram_words, 3, True)
        np2.populate_embeddings_ngrams(ngram_words, 3, True)
        wp1.populate_embeddings(word_words, True)
        wp2.populate_embeddings(word_words, True)
        lp1.populate_embeddings(word_words, True)
        lp2.populate_embeddings(word_words, True)

        return np1.embeddings, wp1.embeddings, lp1.embeddings, np2.embeddings, wp2.embeddings, lp2.embeddings

def get_correlation(model, ngram_words, word_words, f, params):
    f = io.open(f, 'r', encoding='utf-8')
    lines = f.readlines()
    preds = []
    golds = []
    seq1n = []
    seq1w = []
    seq1l = []
    seq2n = []
    seq2w = []
    seq2l = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        if params.combination_type != "ngram-word-lstm":
            nX1, wX1, nX2, wX2 = get_seqs(p1, p2, ngram_words, word_words, params)
        else:
            nX1, wX1, lX1, nX2, wX2, lX2 = get_seqs(p1, p2, ngram_words, word_words, params)
        #pdb.set_trace()
        seq1n.append(nX1)
        seq1w.append(wX1)
        seq2n.append(nX2)
        seq2w.append(wX2)
        if params.combination_type == "ngram-word-lstm":
            seq1l.append(lX1)
            seq2l.append(lX2)
        ct += 1
        if ct % 100 == 0:
            nx1, nm1 = model.prepare_data(seq1n)
            wx1, wm1 = model.prepare_data(seq1w)
            nx2, nm2 = model.prepare_data(seq2n)
            wx2, wm2 = model.prepare_data(seq2w)
            if params.combination_type == "ngram-word-lstm":
                lx1, lm1 = model.prepare_data(seq1l)
                lx2, lm2 = model.prepare_data(seq2l)
            if params.combination_type != "ngram-word-lstm":
                scores = model.scoring_function(nx1,nm1,wx1,wm1,nx2,nm2,wx2,wm2)
            else:
                scores = model.scoring_function(nx1, nm1, wx1, wm1, lx1, lm1,
                                                nx2, nm2, wx2, wm2, lx2, lm2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1n = []
            seq1w = []
            seq2n = []
            seq2w = []
            seq1l = []
            seq2l = []
        golds.append(score)
    if len(seq1n) > 0:
        nx1, nm1 = model.prepare_data(seq1n)
        wx1, wm1 = model.prepare_data(seq1w)
        nx2, nm2 = model.prepare_data(seq2n)
        wx2, wm2 = model.prepare_data(seq2w)
        if params.combination_type == "ngram-word-lstm":
            lx1, lm1 = model.prepare_data(seq1l)
            lx2, lm2 = model.prepare_data(seq2l)
        if params.combination_type != "ngram-word-lstm":
            scores = model.scoring_function(nx1, nm1, wx1, wm1, nx2, nm2, wx2, wm2)
        else:
            scores = model.scoring_function(nx1, nm1, wx1, wm1, lx1, lm1,
                                            nx2, nm2, wx2, wm2, lx2, lm2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def evaluate_all(model, ngram_words, word_words, params):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["annotated-ppdb-dev",
            "annotated-ppdb-test"]

    for i in farr:
        p,s = get_correlation(model, ngram_words, word_words, prefix + i, params)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j,k in zip(parr,sarr,farr):
        s += str(i)+" "+str(j)+" "+k+" | "

    print s
    return parr[0]