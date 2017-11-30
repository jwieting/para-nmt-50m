import cPickle
import copy
import time
import lasagne
import numpy as np
import theano
import utils
from theano import config
from theano import tensor as T
from evaluate_mixed import evaluate_all
from lasagne_layers import lasagne_add_layer
from lasagne_layers import lasagne_average_layer

class mixed_models(object):

    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype=config.floatX)
        return x, x_mask

    def save_params(self, fname, words):
        f = file(fname, 'wb')
        values = lasagne.layers.get_all_param_values(self.final_layer)
        values.append(words)
        cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
            minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    def get_pairs(self, batch, params):

        g1n = []
        g1w = []
        g1l = []
        g2n = []
        g2w = []
        g2l = []

        for i in batch:
            if params.combination_type != "ngram-word-lstm":
                g1n.append(i[0].embeddings)
                g1w.append(i[1].embeddings)
                g2n.append(i[2].embeddings)
                g2w.append(i[3].embeddings)
            else:
                g1n.append(i[0].embeddings)
                g1w.append(i[1].embeddings)
                g1l.append(i[2].embeddings)
                g2n.append(i[3].embeddings)
                g2w.append(i[4].embeddings)
                g2l.append(i[5].embeddings)

        g1nx, g1nmask = self.prepare_data(g1n)
        g1wx, g1wmask = self.prepare_data(g1w)
        g2nx, g2nmask = self.prepare_data(g2n)
        g2wx, g2wmask = self.prepare_data(g2w)
        if params.combination_type == "ngram-word-lstm":
            g1lx, g1lmask = self.prepare_data(g1l)
            g2lx, g2lmask = self.prepare_data(g2l)

        if params.combination_type != "ngram-word-lstm":
            embg1 = self.feedforward_function(g1nx, g1nmask, g1wx, g1wmask)
            embg2 = self.feedforward_function(g2nx, g2nmask, g2wx, g2wmask)
        else:
            embg1 = self.feedforward_function(g1nx, g1nmask, g1wx, g1wmask, g1lx, g1lmask)
            embg2 = self.feedforward_function(g2nx, g2nmask, g2wx, g2wmask, g2lx, g2lmask)

        if params.combination_type != "ngram-word-lstm":
            batch_n = []
            batch_w = []
            for idx, i in enumerate(batch):
                i[0].representation = embg1[idx, :]
                i[1].representation = embg1[idx, :]
                i[2].representation = embg2[idx, :]
                i[3].representation = embg2[idx, :]
                batch_n.append((i[0],i[2]))
                batch_w.append((i[1],i[3]))
        else:
            batch_n = []
            batch_w = []
            batch_l = []
            for idx, i in enumerate(batch):
                i[0].representation = embg1[idx, :]
                i[1].representation = embg1[idx, :]
                i[2].representation = embg1[idx, :]
                i[3].representation = embg2[idx, :]
                i[4].representation = embg2[idx, :]
                i[5].representation = embg2[idx, :]
                batch_n.append((i[0],i[3]))
                batch_w.append((i[1],i[4]))
                batch_l.append((i[2],i[5]))

        pairs1 = utils.get_pairs_fast(batch_n, params.samplingtype)
        p1n = []
        p2n = []
        for i in pairs1:
            p1n.append(i[0].embeddings)
            p2n.append(i[1].embeddings)

        p1nx, p1nmask = self.prepare_data(p1n)
        p2nx, p2nmask = self.prepare_data(p2n)

        pairs2 = utils.get_pairs_fast(batch_w, params.samplingtype)
        p1w = []
        p2w = []
        for i in pairs2:
            p1w.append(i[0].embeddings)
            p2w.append(i[1].embeddings)

        p1wx, p1wmask = self.prepare_data(p1w)
        p2wx, p2wmask = self.prepare_data(p2w)

        if params.combination_type == "ngram-word-lstm":
            pairs3 = utils.get_pairs_fast(batch_l, params.samplingtype)
            p1l = []
            p2l = []
            for i in pairs3:
                p1l.append(i[0].embeddings)
                p2l.append(i[1].embeddings)

            p1lx, p1lmask = self.prepare_data(p1l)
            p2lx, p2lmask = self.prepare_data(p2l)

        if params.combination_type != "ngram-word-lstm":
            return (g1nx, g1nmask, g1wx, g1wmask, g2nx, g2nmask, g2wx, g2wmask,
                p1nx, p1nmask, p1wx, p1wmask, p2nx, p2nmask, p2wx, p2wmask)
        else:
            return (g1nx, g1nmask, g1wx, g1wmask, g1lx, g1lmask, g2nx, g2nmask, g2wx, g2wmask, g2lx, g2lmask,
                p1nx, p1nmask, p1wx, p1wmask, p1lx, p1lmask, p2nx, p2nmask, p2wx, p2wmask, p2lx, p2lmask)

    def __init__(self, We_initial_ngrams, We_initial_words, params, We_initial_lstm = None):

        if We_initial_ngrams is not None:
            We_ngrams = theano.shared(np.asarray(We_initial_ngrams, dtype=config.floatX))
        if We_initial_words is not None:
            We_words = theano.shared(np.asarray(We_initial_words, dtype=config.floatX))
        if We_initial_lstm is not None:
            We_lstm = theano.shared(np.asarray(We_initial_lstm, dtype=config.floatX))

        ng_g1 = T.imatrix()
        ng_g2 = T.imatrix()
        ng_p1 = T.imatrix()
        ng_p2 = T.imatrix()
        ng_g1mask = T.matrix()
        ng_g2mask = T.matrix()
        ng_p1mask = T.matrix()
        ng_p2mask = T.matrix()

        wd_g1 = T.imatrix()
        wd_g2 = T.imatrix()
        wd_p1 = T.imatrix()
        wd_p2 = T.imatrix()
        wd_g1mask = T.matrix()
        wd_g2mask = T.matrix()
        wd_p1mask = T.matrix()
        wd_p2mask = T.matrix()

        lstm_g1batchindices = T.imatrix()
        lstm_g2batchindices = T.imatrix()
        lstm_p1batchindices = T.imatrix()
        lstm_p2batchindices = T.imatrix()
        lstm_g1mask = T.matrix()
        lstm_g2mask = T.matrix()
        lstm_p1mask = T.matrix()
        lstm_p2mask = T.matrix()

        ng_inputs = [ng_g1,
        ng_g2,
        ng_p1,
        ng_p2,
        ng_g1mask,
        ng_g2mask,
        ng_p1mask,
        ng_p2mask]

        wd_inputs = [wd_g1,
        wd_g2,
        wd_p1,
        wd_p2,
        wd_g1mask,
        wd_g2mask,
        wd_p1mask,
        wd_p2mask]

        lstm_inputs = [lstm_g1batchindices,
        lstm_g2batchindices,
        lstm_p1batchindices,
        lstm_p2batchindices,
        lstm_g1mask,
        lstm_g2mask,
        lstm_p1mask,
        lstm_p2mask]

        if "ngram" in params.combination_type:
            l_in_ngrams = lasagne.layers.InputLayer((None, None))
            l_mask_ngrams = lasagne.layers.InputLayer(shape=(None, None))
            l_emb_ngrams = lasagne.layers.EmbeddingLayer(l_in_ngrams, input_size=We_ngrams.get_value().shape[0],
                                                         output_size=We_ngrams.get_value().shape[1], W=We_ngrams)

            l_out_ngrams = lasagne_average_layer([l_emb_ngrams, l_mask_ngrams], tosum=False)

        if "word" in params.combination_type:
            l_in_words = lasagne.layers.InputLayer((None, None))
            l_mask_words = lasagne.layers.InputLayer(shape=(None, None))
            l_emb_words = lasagne.layers.EmbeddingLayer(l_in_words, input_size=We_words.get_value().shape[0],
                                                        output_size=We_words.get_value().shape[1], W=We_words)
            l_out_wd = lasagne_average_layer([l_emb_words, l_mask_words], tosum=False)

        if "lstm" in params.combination_type:
            l_in_lstm = lasagne.layers.InputLayer((None, None))
            l_mask_lstm = lasagne.layers.InputLayer(shape=(None, None))
            l_emb_lstm = lasagne.layers.EmbeddingLayer(l_in_lstm, input_size=We_lstm.get_value().shape[0],
                                                           output_size=We_lstm.get_value().shape[1], W=We_lstm)
            l_lstm = lasagne.layers.LSTMLayer(l_emb_lstm, params.dim, peepholes=True, learn_init=False,
                                              mask_input=l_mask_lstm)
            l_out_lstm = lasagne_average_layer([l_lstm, l_mask_lstm], tosum=False)

        lis = []
        if "ngram" in params.combination_type:
            lis.append(l_out_ngrams)
        if "word" in params.combination_type:
            lis.append(l_out_wd)
        if "lstm" in params.combination_type:
            lis.append(l_out_lstm)

        if params.combination_method == "add":
            l_out = lasagne_add_layer(lis)
        elif params.combination_method == "concat":
            l_out = lasagne.layers.ConcatLayer(lis)

        self.final_layer = l_out

        if params.combination_type == "ngram-word":
            embg1 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_g1, l_mask_ngrams: ng_g1mask, l_in_words: wd_g1, l_mask_words: wd_g1mask})
            embg2 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_g2, l_mask_ngrams: ng_g2mask, l_in_words: wd_g2, l_mask_words: wd_g2mask})
            embp1 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_p1, l_mask_ngrams: ng_p1mask, l_in_words: wd_p1, l_mask_words: wd_p1mask})
            embp2 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_p2, l_mask_ngrams: ng_p2mask, l_in_words: wd_p2, l_mask_words: wd_p2mask})
        elif params.combination_type == "ngram-lstm":
            embg1 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_g1, l_mask_ngrams: ng_g1mask, l_in_lstm: lstm_g1batchindices, l_mask_lstm: lstm_g1mask})
            embg2 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_g2, l_mask_ngrams: ng_g2mask, l_in_lstm: lstm_g2batchindices, l_mask_lstm: lstm_g2mask})
            embp1 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_p1, l_mask_ngrams: ng_p1mask, l_in_lstm: lstm_p1batchindices, l_mask_lstm: lstm_p1mask})
            embp2 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_p2, l_mask_ngrams: ng_p2mask, l_in_lstm: lstm_p2batchindices, l_mask_lstm: lstm_p2mask})
        elif params.combination_type == "word-lstm":
            embg1 = lasagne.layers.get_output(l_out, {l_in_words: wd_g1, l_mask_words: wd_g1mask,
                                                      l_in_lstm: lstm_g1batchindices, l_mask_lstm: lstm_g1mask})
            embg2 = lasagne.layers.get_output(l_out, {l_in_words: wd_g2, l_mask_words: wd_g2mask,
                                                      l_in_lstm: lstm_g2batchindices, l_mask_lstm: lstm_g2mask})
            embp1 = lasagne.layers.get_output(l_out, {l_in_words: wd_p1, l_mask_words: wd_p1mask,
                                                      l_in_lstm: lstm_p1batchindices, l_mask_lstm: lstm_p1mask})
            embp2 = lasagne.layers.get_output(l_out, {l_in_words: wd_p2, l_mask_words: wd_p2mask,
                                                      l_in_lstm: lstm_p2batchindices, l_mask_lstm: lstm_p2mask})
        elif params.combination_type == "ngram-word-lstm":
            embg1 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_g1, l_mask_ngrams: ng_g1mask,
                                                      l_in_words: wd_g1, l_mask_words: wd_g1mask,
                                                      l_in_lstm: lstm_g1batchindices, l_mask_lstm: lstm_g1mask})
            embg2 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_g2, l_mask_ngrams: ng_g2mask,
                                                      l_in_words: wd_g2, l_mask_words: wd_g2mask,
                                                      l_in_lstm: lstm_g2batchindices, l_mask_lstm: lstm_g2mask})
            embp1 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_p1, l_mask_ngrams: ng_p1mask,
                                                      l_in_words: wd_p1, l_mask_words: wd_p1mask,
                                                      l_in_lstm: lstm_p1batchindices, l_mask_lstm: lstm_p1mask})
            embp2 = lasagne.layers.get_output(l_out, {l_in_ngrams: ng_p2, l_mask_ngrams: ng_p2mask,
                                                      l_in_words: wd_p2, l_mask_words: wd_p2mask,
                                                      l_in_lstm: lstm_p2batchindices, l_mask_lstm: lstm_p2mask})

        def fix(x):
            return x*(x > 0) + 1E-10*(x <= 0)

        g1g2 = (embg1 * embg2).sum(axis=1)
        g1g2norm = T.sqrt(fix(T.sum(embg1 ** 2, axis=1))) * T.sqrt(fix(T.sum(embg2 ** 2, axis=1)))
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1 * embg1).sum(axis=1)
        p1g1norm = T.sqrt(fix(T.sum(embp1 ** 2, axis=1))) * T.sqrt(fix(T.sum(embg1 ** 2, axis=1)))
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2 * embg2).sum(axis=1)
        p2g2norm = T.sqrt(fix(T.sum(embp2 ** 2, axis=1))) * T.sqrt(fix(T.sum(embg2 ** 2, axis=1)))
        p2g2 = p2g2 / p2g2norm

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1 * (costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2 * (costp2g2 > 0)

        cost = costp1g1 + costp2g2
        network_params = lasagne.layers.get_all_params(l_out, trainable=True)
        network_params.pop(0)
        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True)
        print self.all_params

        cost = T.mean(cost)

        g1g2 = (embg1 * embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1 ** 2, axis=1)) * T.sqrt(T.sum(embg2 ** 2, axis=1))
        g1g2 = g1g2 / g1g2norm
        prediction = g1g2

        if params.combination_type == "ngram-word":
            self.feedforward_function = theano.function([ng_g1, ng_g1mask,
                                                     wd_g1, wd_g1mask], embg1)
            self.scoring_function = theano.function([ng_g1, ng_g1mask,
                                                     wd_g1, wd_g1mask, ng_g2, ng_g2mask,
                                                     wd_g2, wd_g2mask],prediction)
        elif params.combination_type == "ngram-lstm":
            self.feedforward_function = theano.function([ng_g1, ng_g1mask,
                                                     lstm_g1batchindices, lstm_g1mask], embg1)
            self.scoring_function = theano.function([ng_g1, ng_g1mask,
                                                     lstm_g1batchindices, lstm_g1mask, ng_g2, ng_g2mask,
                                                     lstm_g2batchindices, lstm_g2mask],prediction)
        elif params.combination_type == "word-lstm":
            self.feedforward_function = theano.function([wd_g1, wd_g1mask,
                                                     lstm_g1batchindices, lstm_g1mask], embg1)
            self.scoring_function = theano.function([wd_g1, wd_g1mask,
                                                     lstm_g1batchindices, lstm_g1mask, wd_g2, wd_g2mask,
                                                     lstm_g2batchindices, lstm_g2mask],prediction)
        elif params.combination_type == "ngram-word-lstm":
            self.feedforward_function = theano.function([ng_g1, ng_g1mask,
                                                     wd_g1, wd_g1mask,
                                                     lstm_g1batchindices, lstm_g1mask], embg1)
            self.scoring_function = theano.function([ng_g1, ng_g1mask,
                                                     wd_g1, wd_g1mask,
                                                     lstm_g1batchindices, lstm_g1mask,
                                                     ng_g2, ng_g2mask,
                                                     wd_g2, wd_g2mask,
                                                     lstm_g2batchindices, lstm_g2mask],prediction)

        grads = theano.gradient.grad(cost, self.all_params)
        updates = params.learner(grads, self.all_params, params.eta)

        cost = costp1g1 + costp2g2
        cost = T.mean(cost)

        if params.combination_type == "ngram-word":
            self.train_function = theano.function(ng_inputs + wd_inputs, cost, updates=updates)
            self.cost_function = theano.function(ng_inputs + wd_inputs, cost)
        elif params.combination_type == "ngram-lstm":
            self.train_function = theano.function(ng_inputs + lstm_inputs, cost, updates=updates)
            self.cost_function = theano.function(ng_inputs + lstm_inputs, cost)
        elif params.combination_type == "word-lstm":
            self.train_function = theano.function(wd_inputs + lstm_inputs, cost, updates=updates)
            self.cost_function = theano.function(wd_inputs + lstm_inputs, cost)
        elif params.combination_type == "ngram-word-lstm":
            self.train_function = theano.function(ng_inputs + wd_inputs + lstm_inputs, cost, updates=updates)
            self.cost_function = theano.function(ng_inputs + wd_inputs + lstm_inputs, cost)

        print "Num Params:", lasagne.layers.count_params(self.final_layer)

    def train(self, data, ngram_words, word_words, params):

        start_time = time.time()
        evaluate_all(self, ngram_words, word_words, params)

        old_v = 0
        try:

            for eidx in xrange(params.epochs):

                kf = self.get_minibatches_idx(len(data), params.batchsize, shuffle=True)
                lkf = len(kf)
                uidx = 0

                while(len(kf) > 0):

                    megabatch = []
                    idxs = []
                    idx = 0
                    for i in range(params.mb_batchsize):
                        if len(kf) > 0:
                            arr = [data[t] for t in kf[0][1]]
                            curr_idxs = [i + idx for i in range(len(kf[0][1]))]
                            kf.pop(0)
                            megabatch.extend(arr)
                            idxs.append(curr_idxs)
                            idx += len(curr_idxs)
                    uidx += len(idxs)

                    megabatch2 = []
                    for n,i in enumerate(megabatch):
                        example = (i[0],copy.deepcopy(i[0]),i[1],copy.deepcopy(i[1]))
                        if params.combination_type == "ngram-word-lstm":
                            example = (i[0], copy.deepcopy(i[0]), copy.deepcopy(i[0]), i[1], copy.deepcopy(i[1]), copy.deepcopy(i[1]))
                        if params.combination_type == "ngram-word" or params.combination_type == "ngram-lstm":
                            example[0].populate_embeddings_ngrams(ngram_words, 3, True)
                            example[1].populate_embeddings(word_words, True)
                            example[2].populate_embeddings_ngrams(ngram_words, 3, True)
                            example[3].populate_embeddings(word_words, True)
                        elif params.combination_type == "word-lstm":
                            example[0].populate_embeddings(word_words, True)
                            example[1].populate_embeddings(word_words, True)
                            example[2].populate_embeddings(word_words, True)
                            example[3].populate_embeddings(word_words, True)
                        elif params.combination_type == "ngram-word-lstm":
                            example[0].populate_embeddings_ngrams(ngram_words, 3, True)
                            example[1].populate_embeddings(word_words, True)
                            example[2].populate_embeddings(word_words, True)
                            example[3].populate_embeddings_ngrams(ngram_words, 3, True)
                            example[4].populate_embeddings(word_words, True)
                            example[5].populate_embeddings(word_words, True)
                        megabatch2.append(example)
                    megabatch = megabatch2

                    if params.combination_type != "ngram-word-lstm":
                        (g1nx, g1nmask, g1wx, g1wmask, g2nx, g2nmask, g2wx, g2wmask,
                        p1nx, p1nmask, p1wx, p1wmask, p2nx, p2nmask, p2wx, p2wmask) \
                            = self.get_pairs(megabatch, params)
                    else:
                        (g1nx, g1nmask, g1wx, g1wmask, g1lx, g1lmask, g2nx, g2nmask, g2wx, g2wmask, g2lx, g2lmask,
                        p1nx, p1nmask, p1wx, p1wmask, p1lx, p1lmask, p2nx, p2nmask, p2wx, p2wmask, p2lx, p2lmask) \
                            = self.get_pairs(megabatch, params)

                    cost = 0
                    for i in idxs:
                        if params.combination_type != "ngram-word-lstm":
                            cost += self.train_function(g1nx[i], g2nx[i], p1nx[i], p2nx[i],
                                                    g1nmask[i], g2nmask[i], p1nmask[i], p2nmask[i],
                                                    g1wx[i], g2wx[i], p1wx[i], p2wx[i],
                                                    g1wmask[i], g2wmask[i], p1wmask[i], p2wmask[i])
                        else:
                            cost += self.train_function(g1nx[i], g2nx[i], p1nx[i], p2nx[i],
                                                    g1nmask[i], g2nmask[i], p1nmask[i], p2nmask[i],
                                                    g1wx[i], g2wx[i], p1wx[i], p2wx[i],
                                                    g1wmask[i], g2wmask[i], p1wmask[i], p2wmask[i],
                                                    g1lx[i], g2lx[i], p1lx[i], p2lx[i],
                                                    g1lmask[i], g2lmask[i], p1lmask[i], p2lmask[i])

                    cost = cost / len(idxs)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'

                    if utils.check_if_quarter(uidx-len(idxs), uidx, lkf):
                        if params.evaluate:
                            v = evaluate_all(self, ngram_words, word_words, params)
                        if params.save:
                            if v > old_v:
                                old_v = v
                                self.save_params(params.outfile + '.pickle', (ngram_words, word_words))

                    for i in megabatch:
                        i[0].representation = None
                        i[1].representation = None
                        i[2].representation = None
                        i[3].representation = None
                        if params.combination_type == "ngram-word-lstm":
                            i[4].representation = None
                            i[5].representation = None
                        i[0].unpopulate_embeddings()
                        i[1].unpopulate_embeddings()
                        i[2].unpopulate_embeddings()
                        i[3].unpopulate_embeddings()
                        if params.combination_type == "ngram-word-lstm":
                            i[4].representation = None
                            i[5].representation = None

                if params.evaluate:
                    v = evaluate_all(self, ngram_words, word_words, params)

                if params.save:
                    if v > old_v:
                        old_v = v
                        self.save_params(params.outfile + '.pickle', (ngram_words, word_words))

                print 'Epoch ', (eidx + 1), 'Cost ', cost

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.time()
        print "total time:", (end_time - start_time)
