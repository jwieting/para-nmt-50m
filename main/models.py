import cPickle
import time
import lasagne
import numpy as np
import theano
import utils
from theano import config
from theano import tensor as T
from evaluate import evaluate_all
from lasagne_layers import lasagne_average_layer
from lasagne_layers import lasagne_average_layer2
from lasagne_layers import lasagne_cleanse_layer
from lasagne_layers import lasagne_max_layer
from lasagne_layers import lasagne_max_layer2

class models(object):

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
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    def get_pairs(self, batch, params):
        g1 = []
        g2 = []

        for i in batch:
            g1.append(i[0].embeddings)
            g2.append(i[1].embeddings)

        g1_ = [g1[i:i + params.batchsize] for i in xrange(0, len(g1), params.batchsize)]
        g2_ = [g2[i:i + params.batchsize] for i in xrange(0, len(g2), params.batchsize)]
        embg1 = []
        embg2 = []
        for i in range(len(g1_)):
            g1x, g1mask = self.prepare_data(g1_[i])
            g2x, g2mask = self.prepare_data(g2_[i])

            embg1_ = self.feedforward_function(g1x, g1mask)
            embg2_ = self.feedforward_function(g2x, g2mask)
            embg1.append(embg1_)
            embg2.append(embg2_)
        embg1 = np.vstack(embg1)
        embg2 = np.vstack(embg2)

        #update representations
        for idx, i in enumerate(batch):
            i[0].representation = embg1[idx, :]
            i[1].representation = embg2[idx, :]

        pairs = utils.get_pairs_fast(batch, params.samplingtype)
        p1 = []
        p2 = []
        for i in pairs:
            p1.append(i[0].embeddings)
            p2.append(i[1].embeddings)

        p1x, p1mask = self.prepare_data(p1)
        p2x, p2mask = self.prepare_data(p2)

        g1x, g1mask = self.prepare_data(g1)
        g2x, g2mask = self.prepare_data(g2)
        return (g1x, g1mask, g2x, g2mask, p1x, p1mask, p2x, p2mask)

    def __init__(self, We_initial, params):

        initial_We = theano.shared(np.asarray(We_initial, dtype=config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype=config.floatX))

        g1 = T.imatrix()
        g2 = T.imatrix()
        p1 = T.imatrix()
        p2 = T.imatrix()
        g1mask = T.matrix()
        g2mask = T.matrix()
        p1mask = T.matrix()
        p2mask = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0],
                                              output_size=We.get_value().shape[1], W=We)

        if params.model == "wordaverage":
            l_out = lasagne_average_layer([l_emb, l_mask], tosum=False)

        elif params.model == "maxpool":
            l_out = lasagne_max_layer([l_emb, l_mask], params)

        elif params.model == "lstmavg":
            l_lstm = lasagne.layers.LSTMLayer(l_emb, params.dim, peepholes=True, learn_init=False,
                                              mask_input=l_mask)
            l_out = lasagne_average_layer([l_lstm, l_mask], tosum = False)

        elif params.model == "lstmmax":
            l_lstm = lasagne.layers.LSTMLayer(l_emb, params.dim, peepholes=True, learn_init=False,
                                              mask_input=l_mask)
            l_out = lasagne_max_layer([l_lstm, l_mask], params)

        elif params.model == "bilstmavg":
            l_lstm = lasagne.layers.LSTMLayer(l_emb, params.dim, peepholes=True, learn_init=False,
                                              mask_input=l_mask)
            l_lstmb = lasagne.layers.LSTMLayer(l_emb, params.dim, learn_init=False,
                                               mask_input=l_mask, backwards=True)

            l_cleanse = lasagne_cleanse_layer([l_lstm, l_mask], to_pool = False)
            l_cleanse_b = lasagne_cleanse_layer([l_lstmb, l_mask], to_pool = False)
            l_concat = lasagne.layers.ConcatLayer([l_cleanse, l_cleanse_b], axis = params.axis)
            l_out = lasagne_average_layer2([l_concat, l_mask])

        elif params.model == "bilstmmax":
            l_lstm = lasagne.layers.LSTMLayer(l_emb, params.dim, peepholes=True, learn_init=False,
                                              mask_input=l_mask)
            l_lstmb = lasagne.layers.LSTMLayer(l_emb, params.dim, learn_init=False,
                                               mask_input=l_mask, backwards=True)
            l_cleanse = lasagne_cleanse_layer([l_lstm, l_mask], to_pool = True)
            l_cleanse_b = lasagne_cleanse_layer([l_lstmb, l_mask], to_pool = True)
            l_concat = lasagne.layers.ConcatLayer([l_cleanse, l_cleanse_b], axis=params.axis)
            l_out = lasagne_max_layer2([l_concat])

        embg1 = lasagne.layers.get_output(l_out, {l_in: g1, l_mask: g1mask}, deterministic=False)
        embg2 = lasagne.layers.get_output(l_out, {l_in: g2, l_mask: g2mask}, deterministic=False)
        embp1 = lasagne.layers.get_output(l_out, {l_in: p1, l_mask: p1mask}, deterministic=False)
        embp2 = lasagne.layers.get_output(l_out, {l_in: p2, l_mask: p2mask}, deterministic=False)

        def fix(x):
            return x*(x > 0) + 1E-10*(x <= 0)

        #objective function
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

        self.final_layer = l_out
        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True)

        if params.LC:
            l2 = 0.5 * params.LC * sum(lasagne.regularization.l2(x) for x in network_params)
        else:
            l2 = 0
        word_reg = 0.5 * params.LW * lasagne.regularization.l2(We - initial_We)
        cost = T.mean(cost) + l2 + word_reg

        self.feedforward_function = theano.function([g1, g1mask], embg1)

        prediction = g1g2
        self.scoring_function = theano.function([g1, g2,
            g1mask, g2mask],prediction)

        grads = theano.gradient.grad(cost, self.all_params)
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1, g2, p1, p2,
                                                  g1mask, g2mask, p1mask, p2mask], cost, updates=updates)

        cost = costp1g1 + costp2g2
        cost = T.mean(cost)
        self.cost_function = theano.function([g1, g2, p1, p2,
                                                  g1mask, g2mask, p1mask, p2mask], cost)

        print "Num Params:", lasagne.layers.count_params(self.final_layer)

    def train(self, data, words, params):

        start_time = time.time()
        evaluate_all(self, words, params)

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

                    for i in megabatch:
                        if params.wordtype == "words":
                            if params.scramble > 0:
                                n = np.random.binomial(1, params.scramble, 1)[0]
                                if n > 0:
                                    i[0].populate_embeddings_scramble(words)
                                    i[1].populate_embeddings_scramble(words)
                                else:
                                    i[0].populate_embeddings(words, True)
                                    i[1].populate_embeddings(words, True)
                            else:
                                i[0].populate_embeddings(words, True)
                                i[1].populate_embeddings(words, True)
                        else:
                            i[0].populate_embeddings_ngrams(words, 3, True)
                            i[1].populate_embeddings_ngrams(words, 3, True)

                    (g1x, g1mask, g2x, g2mask, p1x, p1mask, p2x, p2mask) = self.get_pairs(megabatch, params)

                    cost = 0
                    for i in idxs:
                        cost += self.train_function(g1x[i], g2x[i], p1x[i], p2x[i], g1mask[i], g2mask[i], p1mask[i], p2mask[i])

                    cost = cost / len(idxs)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'

                    if utils.check_if_quarter(uidx-len(idxs), uidx, lkf):
                        if params.evaluate:
                            v = evaluate_all(self, words, params)
                        if params.save:
                            if v > old_v:
                                old_v = v
                                self.save_params(params.outfile + '.pickle', words)

                    for i in megabatch:
                        i[0].representation = None
                        i[1].representation = None
                        i[0].unpopulate_embeddings()
                        i[1].unpopulate_embeddings()

                if params.evaluate:
                    v = evaluate_all(self, words, params)

                if params.save:
                    if v > old_v:
                        old_v = v
                        self.save_params(params.outfile + '.pickle', words)

                print 'Epoch ', (eidx + 1), 'Cost ', cost

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.time()
        print "total time:", (end_time - start_time)
