import cPickle
import copy
import random
import sys
import argparse
import lasagne
import numpy as np
import utils
from mixed_models import mixed_models
from models import models

def str2learner(v):
    if v is None:
        return None
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not.')

random.seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()

parser.add_argument("-LC", help="Regularization on composition parameters", type=float, default=0.)
parser.add_argument("-LW", help="Regularization on embedding parameters", type=float, default=0.)
parser.add_argument("-outfile", help="Name of output file")
parser.add_argument("-batchsize", help="Size of batch", type=int, default=100)
parser.add_argument("-dim", help="Dimension of model", type=int, default=300)
parser.add_argument("-wordfile", help="Word embedding file", default='../data/paragram_sl999_czeng.txt')
parser.add_argument("-save", help="Whether to pickle model", type=int, default=0)
parser.add_argument("-margin", help="Margin in objective function", type=float, default=0.4)
parser.add_argument("-samplingtype", help="Type of Sampling used: MAX, MIX, or RAND", default="MAX")
parser.add_argument("-evaluate", help="Whether to evaluate the model during training", type=int, default=1)
parser.add_argument("-epochs", help="Number of epochs in training", type=int, default=10)
parser.add_argument("-eta", help="Learning rate", type=float, default=0.001)
parser.add_argument("-learner", help="Either AdaGrad or Adam", default="adam")
parser.add_argument("-model", help="Which model to use between wordaverage, maxpool, (bi)lstmavg, (bi)lstmmax")
parser.add_argument("-scramble", type=float, help="Rate of scrambling", default=0.3)
parser.add_argument("-max", type=int, help="Maximum number of examples to use (<= 0 means use all data)", default=0)
parser.add_argument("-loadmodel", help="Name of pickle file containing model", default=None)
parser.add_argument("-data", help="Name of data file containing paraphrases", default=None)
parser.add_argument("-wordtype", help="Either words or 3grams", default="words")
parser.add_argument("-random_embs", help="Whether to use random embeddings "
                                         "and not pretrained embeddings", type = int, default=0)
parser.add_argument("-mb_batchsize", help="Size of megabatch", type=int, default=40)
parser.add_argument("-axis", help="Axis on which to concatenate hidden "
                                  "states (1 for sequence, 2 for embeddings)", type=int, default=1)
parser.add_argument("-combination_method", help="Type of combining models (either add or concat)")
parser.add_argument("-combination_type", help="choices are ngram-word, ngram-lstm, "
                                        "ngram-word-lstm, word-lstm")

args = parser.parse_args()
args.learner = str2learner(args.learner)
print " ".join(sys.argv)

params = args

data = utils.get_data(params.data)

if params.combination_type:
    if params.loadmodel:
        saved_params = cPickle.load(open(params.loadmodel, 'rb'))
        if params.combination_type == "ngram-word":
            words = saved_params.pop(-1)
            words_3grams = words[0]
            words_words = words[1]
        elif params.combination_type == "ngram-word-lstm":
            words = saved_params.pop(-1)
            words_3grams = words[0]
            words_words = words[1]

        if params.combination_type == "ngram-word":
            model = mixed_models(saved_params[0], saved_params[1], params)
        elif params.combination_type == "ngram-word-lstm":
            model = mixed_models(saved_params[0], saved_params[1], params, We_initial_lstm = saved_params[2])

        lasagne.layers.set_all_param_values(model.final_layer, saved_params)
    else:
        words_3grams, We_3gram = utils.get_ngrams(data, params)
        if params.random_embs:
            words_words, We_word = utils.get_words(data, params)
        else:
            words_words, We_word = utils.get_wordmap(args.wordfile)

        We_lstm = copy.deepcopy(We_word)

        if params.combination_type == "ngram-word":
            model = mixed_models(We_3gram, We_word, params)
        elif params.combination_type == "ngram-lstm":
            model = mixed_models(We_3gram, None, params, We_initial_lstm = We_lstm)
        elif params.combination_type == "word-lstm":
            model = mixed_models(None, We_word, params, We_initial_lstm = We_lstm)
        elif params.combination_type == "ngram-word-lstm":
            model = mixed_models(We_3gram, We_word, params, We_initial_lstm = We_lstm)
        else:
            print "Please enter a valid combination type. Exiting."
            sys.exit(0)

    print "Num examples:", len(data)
    print "Num n-grams:", len(words_3grams)
    print "Num words:", len(words_words)

    model.train(data, words_3grams, words_words, params)
else:
    if params.loadmodel:
        saved_params = cPickle.load(open(params.loadmodel, 'rb'))
        words = saved_params.pop(-1)
        model = models(saved_params[0], params)
        lasagne.layers.set_all_param_values(model.final_layer, saved_params)
    else:
        if params.wordtype == "words":
            if params.random_embs:
                words, We = utils.get_words(data, params)
            else:
                words, We = utils.get_wordmap(args.wordfile)
        else:
            words, We = utils.get_ngrams(data, params)

        model = models(We, params)

    print "Num examples:", len(data)
    print "Num words:", len(words)

    model.train(data, words, params)