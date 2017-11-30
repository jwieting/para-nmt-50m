import lasagne
from theano import tensor as T

class lasagne_average_layer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, tosum=False, **kwargs):
        super(lasagne_average_layer, self).__init__(incoming, **kwargs)
        self.tosum = tosum

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        mask = inputs[1]
        emb = (emb * mask[:, :, None]).sum(axis=1)
        if not self.tosum:
            emb = emb / mask.sum(axis=1)[:, None]
        return emb

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

class lasagne_average_layer2(lasagne.layers.MergeLayer):

    def __init__(self, incoming, **kwargs):
        super(lasagne_average_layer2, self).__init__(incoming, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        mask = inputs[1]
        emb = emb.sum(axis=1)
        emb = emb / mask.sum(axis=1)[:, None]
        return emb

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

class lasagne_max_layer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, params, **kwargs):
        super(lasagne_max_layer, self).__init__(incoming, **kwargs)
        self.dim = params.dim

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        mask = inputs[1]
        emb = (emb * mask[:, :, None])
        emb += ((mask - 1) * 100)[:, :, None]
        return T.max(emb, axis=1)

class lasagne_max_layer2(lasagne.layers.MergeLayer):

    def __init__(self, incoming, **kwargs):
        super(lasagne_max_layer2, self).__init__(incoming, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        return T.max(emb, axis=1)

class lasagne_add_layer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, tosum=False, **kwargs):
        super(lasagne_add_layer, self).__init__(incoming, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        emb1 = inputs[0]
        emb2 = inputs[1]
        if len(inputs) == 3:
            emb3 = inputs[2]
            return emb1 + emb2 + emb3
        return emb1 + emb2

    def get_output_shape_for(self, input_shape):
        return input_shape

class lasagne_cleanse_layer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, to_pool = False, **kwargs):
        super(lasagne_cleanse_layer, self).__init__(incoming, **kwargs)
        self.to_pool = to_pool

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        mask = inputs[1]
        emb = (emb * mask[:, :, None])
        if self.to_pool:
            emb += ((mask - 1) * 100)[:, :, None]
        return emb

    def get_output_shape_for(self, input_shape):
        return input_shape[0]