# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Point cloud models (utility functions).
"""

import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate


class CustomConv(Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 normalization=False,
                 activation=None,
                 **kwargs):
        super(CustomConv, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.normalization = normalization
        self.activation = activation

        # define layers
        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=not self.normalization)
        if self.normalization:
            self.norm = LayerNormalization()
        else:
            self.norm = None
        self.act = Activation(
            activation=self.activation)

    def call(self, inputs, *args, **kwargs):
        # compute layers
        x = self.conv(inputs)
        if self.normalization:
            x = self.norm(x)
        x = self.act(x)

        return x

    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'normalization': self.normalization,
            'activation': self.activation})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BlockConv(Layer):
    def __init__(self,
                 l_filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 normalization=False,
                 activation=None,
                 **kwargs):
        super(BlockConv, self).__init__(**kwargs)

        # initialize parameters
        self.l_filters = l_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.normalization = normalization
        self.activation = activation

        # define layers
        self.layers = []
        for filters in self.l_filters:
            layer = CustomConv(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                normalization=self.normalization,
                activation=self.activation)
            self.layers.append(layer)

    def call(self, inputs, *args, **kwargs):
        # get inputs
        x = inputs

        # compute layers
        for layer in self.layers:
            x = layer(x)

        return x

    def get_config(self):
        config = super(BlockConv, self).get_config()
        config.update({
            'l_filters': self.l_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'normalization': self.normalization,
            'activation': self.activation})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomDense(Layer):

    def __init__(self,
                 units,
                 normalization=False,
                 activation=None,
                 **kwargs):
        super(CustomDense, self).__init__(**kwargs)

        # initialize parameters
        self.units = units
        self.normalization = normalization
        self.activation = activation

        # define layers
        self.dense = Dense(
            units=self.units,
            use_bias=not self.normalization)
        if self.normalization:
            self.norm = LayerNormalization()
        else:
            self.norm = None
        self.act = Activation(
            activation=self.activation)

    def call(self, inputs, *args, **kwargs):
        # compute layers
        x = self.dense(inputs)
        if self.normalization:
            x = self.norm(x)
        x = self.act(x)

        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'normalization': self.normalization,
            'activation': self.activation})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BlockDense(Layer):
    def __init__(self,
                 l_units,
                 normalization=False,
                 activation=None,
                 dropout_rate=0,
                 **kwargs):
        super(BlockDense, self).__init__(**kwargs)

        # initialize parameters
        self.l_units = l_units
        self.normalization = normalization
        self.activation = activation
        self.dropout_rate = dropout_rate

        # define layers
        self.layers = []
        for units in self.l_units:
            layer = CustomDense(
                units=units,
                normalization=self.normalization,
                activation=self.activation)
            self.layers.append(layer)
            if self.dropout_rate > 0:
                self.layers.append(Dropout(rate=self.dropout_rate))

    def call(self, inputs, *args, **kwargs):
        # get inputs
        x = inputs

        # compute layers
        for layer in self.layers:
            x = layer(x)

        return x

    def get_config(self):
        config = super(BlockDense, self).get_config()
        config.update({
            'l_units': self.l_units,
            'normalization': self.normalization,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TNet(Layer):

    def __init__(self,
                 add_regularization=False,
                 **kwargs):
        super(TNet, self).__init__(**kwargs)

        # initialize parameters
        self.add_regularization = add_regularization

        # define layers
        self.block_0 = BlockConv(
            l_filters=[64, 128, 256, 512],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            normalization=True,
            activation="relu")
        self.maxpool = GlobalMaxPool2D()
        self.block_1 = BlockDense(
            l_units=[256, 128],
            normalization=True,
            activation="relu")

        # layers to build
        self.c = None
        self.w = None
        self.eye = None
        self.b = None

    def build(self, input_shape):
        # get number of features
        self.c = input_shape[-1]

        # define a weight matrix and bias
        self.w = self.add_weight(
            shape=(128, self.c ** 2),
            initializer=tf.zeros_initializer,
            trainable=True,
            name="w")

        # define a bias matrix
        def custom_bias_init(shape, dtype=tf.float32):
            return tf.eye(shape[-1], dtype=dtype)
        self.b = self.add_weight(
            shape=(self.c, self.c),
            initializer=custom_bias_init,
            trainable=True,
            name="b")

        # initialize identity matrix
        self.eye = tf.eye(self.c, dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        # compute convolutions
        x = self.block_0(inputs)  # (B, length, 1, 512)

        # compute global features
        x = self.maxpool(x)  # (B, 512)

        # compute fully connected layers
        x = self.block_1(x)  # (B, 128)

        # convert to a CxC matrix
        x = tf.expand_dims(x, axis=1)  # (B, 1, 128)
        x = tf.matmul(x, self.w)  # (B, 1, C*C)
        x = tf.squeeze(x, axis=1)  # (B, C*C)
        x = tf.reshape(x, (-1, self.c, self.c))  # (B, C, C)

        # add bias term (initialized to identity matrix)
        x = tf.add(x, self.b)

        # add regularization
        if self.add_regularization:
            x_xt = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            diff = tf.subtract(self.eye, x_xt)
            reg_loss = tf.nn.l2_loss(diff)
            reg_loss = tf.scalar_mul(1e-3, reg_loss)
            self.add_loss(reg_loss)

        return x

    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TNetEdge(Layer):

    def __init__(self,
                 k,
                 add_regularization=False,
                 **kwargs):
        super(TNetEdge, self).__init__(**kwargs)

        # initialize parameters
        self.k = k
        self.add_regularization = add_regularization

        # define layers
        self.block_0_edge = BlockEdge(
            k=self.k,
            l_filters=[64, 64, 64],
            activation="relu",
            normalization=True)
        self.block_0 = BlockConv(
            l_filters=[512],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            normalization=True,
            activation="relu")
        self.maxpool = GlobalMaxPool2D()
        self.block_1 = BlockDense(
            l_units=[256, 128],
            normalization=True,
            activation="relu")

        # layers to build
        self.c = None
        self.w = None
        self.eye = None
        self.b = None

    def build(self, input_shape):
        # get number of features
        self.c = input_shape[-1]

        # define a weight matrix and bias
        self.w = self.add_weight(
            shape=(128, self.c ** 2),
            initializer=tf.zeros_initializer,
            trainable=True,
            name="w")

        # define a bias matrix
        def custom_bias_init(shape, dtype=tf.float32):
            return tf.eye(shape[-1], dtype=dtype)
        self.b = self.add_weight(
            shape=(self.c, self.c),
            initializer=custom_bias_init,
            trainable=True,
            name="b")

        # initialize identity matrix
        self.eye = tf.eye(self.c, dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):  # (B, N, C)
        # compute edge convolution
        x = self.block_0_edge(inputs)  # (B, N, 64)
        x = tf.expand_dims(x, axis=2)  # (B, N, 1, 64)

        # compute convolutions
        x = self.block_0(x)  # (B, N, 1, 512)

        # compute global features
        x = self.maxpool(x)  # (B, 512)

        # compute fully connected layers
        x = self.block_1(x)  # (B, 128)

        # convert to a CxC matrix
        x = tf.expand_dims(x, axis=1)  # (B, 1, 128)
        x = tf.matmul(x, self.w)  # (B, 1, C * C)
        x = tf.squeeze(x, axis=1)  # (B, C * C)
        x = tf.reshape(x, (-1, self.c, self.c))  # (B, C, C)

        # add bias term (initialized to identity matrix)
        x = tf.add(x, self.b)

        # add regularization
        if self.add_regularization:
            x_xt = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            diff = tf.subtract(self.eye, x_xt)
            reg_loss = tf.nn.l2_loss(diff)
            reg_loss = tf.scalar_mul(1e-3, reg_loss)
            self.add_loss(reg_loss)

        return x

    def get_config(self):
        config = super(TNetEdge, self).get_config()
        config.update({
            'k': self.k,
            'add_regularization': self.add_regularization})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_additional_features(
        x,
        filters,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        downsampling_indices=None):
    # collect additional features
    additional_features = []
    additional_features_flag = False
    if input_cluster is not None:
        additional_features.append(input_cluster)
    if input_morphology is not None:
        additional_features.append(input_morphology)
    if input_distance is not None:
        additional_features.append(input_distance)

    # concatenate additional features
    if len(additional_features) > 1:
        additional_features = Concatenate(
            axis=-1)(additional_features)  # (B, N, [3-5])
        additional_features = tf.expand_dims(
            additional_features,
            axis=2)  # (B, N, 1, [3-5])
        additional_features_flag = True
    elif len(additional_features) == 1:
        additional_features = additional_features[0]  # (B, N, [1-2])
        additional_features = tf.expand_dims(
            additional_features,
            axis=2)  # (B, N, 1, [1-2])
        additional_features_flag = True

    # MLP block
    if additional_features_flag:
        if downsampling_indices is not None:
            for downsampling_index in downsampling_indices:
                additional_features = tf.gather(
                    additional_features,
                    indices=downsampling_index,
                    axis=1,
                    batch_dims=1)  # (B, N/2, 1, 16)
        block_extra = BlockConv(
            l_filters=filters,
            kernel_size=(1, 1),
            normalization=True,
            activation="relu",
            name="additional_features_block")(additional_features)
        output = Concatenate(
            axis=-1)([x, block_extra])
    else:
        output = x

    return output


def compute_pairwise_distance(inputs):
    # compute pairwise distance
    x_t = tf.transpose(inputs, perm=[0, 2, 1])  # (B, C, N)
    x_inner = tf.matmul(inputs, x_t)
    x_inner = tf.scalar_mul(-2.0, x_inner)  # (B, N, N)
    x_square = tf.square(inputs)
    x_square = tf.reduce_sum(x_square, axis=-1, keepdims=True)  # (B, N, C)
    x_square_t = tf.transpose(x_square, perm=[0, 2, 1])  # (B, C, N)
    x = tf.add(tf.add(x_square, x_inner), x_square_t)  # (B, N, N)

    return x


def get_knn(adjacency_matrix, k):
    x = tf.scalar_mul(-1.0, adjacency_matrix)
    _, nn_idx = tf.nn.top_k(x, k=k)

    return nn_idx


class EdgeFeature(Layer):

    def __init__(self,
                 k,
                 return_edge_only=False,
                 return_knn=False,
                 **kwargs):
        super(EdgeFeature, self).__init__(**kwargs)

        # initialize parameter
        self.k = k
        self.return_edge_only = return_edge_only
        self.return_knn = return_knn

    def call(self, inputs, *args, **kwargs):
        # get tensor shape
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        n_dim = tf.shape(inputs)[2]

        # compute pairwise distances
        pairwise_distance = compute_pairwise_distance(inputs)  # (B, N, N)

        # get k nearest neighbors
        nn_idx = get_knn(
            adjacency_matrix=pairwise_distance,
            k=self.k)  # (B, N, k)

        # get neighbors features
        x_flat = tf.reshape(inputs, [-1, n_dim])  # (B * N, C)
        nn_idx_flat = tf.scalar_mul(length, tf.range(batch_size))
        nn_idx_flat = tf.reshape(nn_idx_flat, [batch_size, 1, 1])  # (B, 1, 1)
        nn_idx_flat = tf.add(nn_idx_flat, nn_idx)  # (B, N, k)
        x_neighbors = tf.gather(x_flat, nn_idx_flat)  # (B, N, k, C)

        # duplicate target feature
        x_target = tf.expand_dims(inputs, axis=2)  # (B, N, 1, C)
        x_target = tf.tile(x_target, [1, 1, self.k, 1])  # (B, N, k, C)

        # format edge features
        x_edge = tf.subtract(x_neighbors, x_target)  # (B, N, k, C)
        if not self.return_edge_only:
            x_edge = tf.concat([x_target, x_edge], axis=-1)  # (B, N, k, 2 * C)

        # return knn
        if self.return_knn:
            return x_edge, nn_idx
        else:
            return x_edge

    def get_config(self):
        config = super(EdgeFeature, self).get_config()
        config.update({
            'k': self.k,
            'return_edge_only': self.return_edge_only,
            'return_knn': self.return_knn})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EdgeConv(Layer):

    def __init__(self,
                 k,
                 filters,
                 activation,
                 add_strides=False,
                 normalization=False,
                 **kwargs):
        super(EdgeConv, self).__init__(**kwargs)

        # initialize parameter
        self.k = k
        self.filters = filters
        self.activation = activation
        self.add_strides = add_strides
        self.normalization = normalization

        # define layer
        self.feature_extractor = EdgeFeature(
            k=self.k)
        if self.add_strides:
            self.conv = CustomConv(
               filters=self.filters,
               kernel_size=(1, self.k),
               strides=(1, self.k),
               normalization=self.normalization,
               activation=self.activation)
        else:
            self.conv = CustomConv(
                filters=self.filters,
                kernel_size=(1, 1),
                normalization=self.normalization,
                activation=self.activation)

    def call(self, inputs, *args, **kwargs):
        # get edge features (B, N, k, 2 * C)
        x = self.feature_extractor(inputs)

        # convolution (and pooling)
        if self.add_strides:
            x = self.conv(x)  # (B, N, 1, filters)
            x = tf.squeeze(x, axis=2)  # (B, N, filters)
        else:
            x = self.conv(x)  # (B, N, k, filters)
            x = tf.reduce_max(x, axis=2)  # (B, N, filters)

        return x

    def get_config(self):
        config = super(EdgeConv, self).get_config()
        config.update({
            'k': self.k,
            'filters': self.filters,
            'activation': self.activation,
            'add_strides': self.add_strides,
            'normalization': self.normalization})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BlockEdge(Layer):

    def __init__(self,
                 k,
                 l_filters,
                 activation=None,
                 add_strides=False,
                 normalization=False,
                 **kwargs):
        super(BlockEdge, self).__init__(**kwargs)

        # initialize parameter
        self.k = k
        self.l_filters = l_filters
        self.activation = activation
        self.add_strides = add_strides
        self.normalization = normalization

        # define layers
        self.layers = []
        for filters in self.l_filters:
            layer = EdgeConv(
                k=self.k,
                filters=filters,
                activation=self.activation,
                add_strides=self.add_strides,
                normalization=self.normalization)
            self.layers.append(layer)

    def call(self, inputs, *args, **kwargs):
        # get inputs
        x = inputs

        # compute layers
        for layer in self.layers:
            x = layer(x)

        return x

    def get_config(self):
        config = super(BlockEdge, self).get_config()
        config.update({
            'k': self.k,
            'l_filters': self.l_filters,
            'activation': self.activation,
            'add_strides': self.add_strides,
            'normalization': self.normalization})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PositionEmbedding(Layer):

    def __init__(self,
                 units,
                 latent_position_units,
                 k,
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)

        # initialize parameter
        self.units = units
        self.latent_position_units = latent_position_units
        self.k = k

        # define layers
        self.edge_extractor = EdgeFeature(
            k=self.k,
            return_edge_only=True,
            return_knn=True)
        self.position_0 = tf.keras.layers.Dense(
            units=self.latent_position_units,
            activation="relu",
            use_bias=False)
        self.position_1 = tf.keras.layers.Dense(
            units=self.units,
            use_bias=False)

    def call(self, inputs, *args, **kwargs):  # (B, N, C)
        # compute edges and get knn (B, N, k, C), (B, N, k)
        edges, nn_idx = self.edge_extractor(inputs)

        # local position embedding (B, N, k, units)
        position_embedding = self.position_0(edges)
        position_embedding = self.position_1(position_embedding)

        return position_embedding, nn_idx

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({
            'units': self.units,
            'latent_position_units': self.latent_position_units,
            'k': self.k})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def compute_query_key_relation(query, key, nn_idx):
    # query  (B, N, units)
    # key  (B, N, units)
    # nn_idx (B, N, k)

    # get tensor shape
    batch_size = tf.shape(query)[0]
    length = tf.shape(query)[1]
    n_dim = tf.shape(query)[2]
    k = tf.shape(nn_idx)[2]

    # get target query
    query_target = tf.expand_dims(query, axis=2)  # (B, N, 1, units)
    query_target = tf.tile(query_target, [1, 1, k, 1])  # (B, N, k, units)

    # get neighbors key
    key_neighbors = tf.reshape(key, [-1, n_dim])  # (B * N, units)
    nn_idx_flat = tf.scalar_mul(length, tf.range(batch_size))
    nn_idx_flat = tf.reshape(nn_idx_flat, [batch_size, 1, 1])  # (B, 1, 1)
    nn_idx_flat = tf.add(nn_idx_flat, nn_idx)  # (B, N, k)
    key_neighbors = tf.gather(key_neighbors, nn_idx_flat)  # (B, N, k, unit)

    # compute relation
    qk = tf.subtract(query_target, key_neighbors)  # (B, N, k, units)

    return qk


class AttentionLayer(Layer):

    def __init__(self,
                 units,
                 latent_units,
                 **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

        # initialize parameter
        self.units = units
        self.latent_units = latent_units

        # define layers
        self.query = Dense(
            units=self.units,
            use_bias=False)
        self.key = Dense(
            units=self.units,
            use_bias=False)
        self.value = Dense(
            units=self.units,
            use_bias=False)
        self.attention_0 = Dense(
            units=self.latent_units,
            activation="relu",
            use_bias=False)
        self.attention_1 = Dense(
            units=self.units,
            use_bias=False)

    def call(self, inputs, *args, **kwargs):
        # get inputs (B, N, C), (B, N, k, C_position), (B, N, k)
        x, position_embedding, nn_idx = inputs

        # linear projections (query, key, value)
        query = self.query(x)  # (B, N, units)
        key = self.key(x)  # (B, N, units)
        value = self.value(x)  # (B, N, units)

        # compute relative weight (x_query - x_key)
        qk = compute_query_key_relation(query, key, nn_idx)  # (B, N, k, units)

        # get neighbors value (B, N, k, units)
        value = tf.gather(value, nn_idx, axis=-2, batch_dims=1)

        # attention layer
        attention = tf.add(qk, position_embedding)  # (B, N, k, units)
        attention = self.attention_0(attention)  # (B, N, k, latent_units)
        attention = self.attention_1(attention)  # (B, N, k, units)
        attention = tf.nn.softmax(attention, axis=-2)  # (B, N, k, units)

        # output layer
        value = tf.add(value, position_embedding)  # (B, N, k, units)
        output = tf.multiply(value, attention)  # (B, N, k, units)
        output = tf.math.reduce_sum(output, axis=-2)  # (B, N, units)

        return output

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'units': self.units,
            'latent_units': self.latent_units})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiAttentionLayer(Layer):

    def __init__(self,
                 units,
                 latent_units,
                 k,
                 nb_head,
                 add_shortcut=False,
                 normalization=False,
                 return_heads=False,
                 **kwargs):
        super(MultiAttentionLayer, self).__init__(**kwargs)

        # initialize parameter
        self.units = units
        self.latent_units = latent_units
        self.k = k
        self.h = nb_head
        self.add_shortcut = add_shortcut
        self.normalization = normalization
        self.return_heads = return_heads

        # define layers
        self.position_embedding = PositionEmbedding(
            units=self.units,
            latent_position_units=self.latent_units,
            k=self.k)
        self.linear_in = Dense(
            units=self.units,
            use_bias=False)
        self.query = Dense(
            units=self.units * self.h,
            use_bias=False)
        self.key = Dense(
            units=self.units * self.h,
            use_bias=False)
        self.value = Dense(
            units=self.units * self.h,
            use_bias=False)
        self.layers = []
        for _ in range(self.h):
            attention_0 = Conv2D(
                filters=self.latent_units,
                kernel_size=(1, 1),
                activation="relu",
                use_bias=False)
            attention_1 = Conv2D(
                filters=self.units,
                kernel_size=(1, 1),
                use_bias=False)
            self.layers.append((attention_0, attention_1))
        self.linear_out = Dense(
            units=self.units,
            use_bias=False)
        if self.normalization:
            self.norm = LayerNormalization()
        else:
            self.norm = None

    def call(self, inputs, *args, **kwargs):
        # inputs (B, N, C)

        # position embedding (B, N, k, h * units), (B, N, k)
        position, nn_idx = self.position_embedding(inputs)
        position = tf.tile(position, [1, 1, 1, self.h])

        # linear projection (B, N, units)
        x = self.linear_in(inputs)

        # linear projections (query, key, value)
        query = self.query(x)  # (B, N, h * units)
        key = self.key(x)  # (B, N, h * units)
        value = self.value(x)  # (B, N, h * units)

        # compute relative weight (x_query - x_key) (B, N, k, h * units)
        qk = compute_query_key_relation(query, key, nn_idx)

        # get neighbors value (B, N, k, h * units)
        value = tf.gather(value, nn_idx, axis=-2, batch_dims=1)

        # attention layer
        attentions = tf.add(qk, position)  # (B, N, k, h * units)
        attentions = tf.split(attentions, self.h, axis=-1)
        attention = []
        for attention_, (layer_0, layer_1) in zip(attentions, self.layers):
            attention_ = layer_0(attention_)  # (B, N, k, latent_units)
            attention_ = layer_1(attention_)  # (B, N, k, units)
            attention.append(attention_)
        attention = tf.concat(attention, axis=-1)  # (B, N, k, h * units)
        attention = tf.nn.softmax(attention, axis=-2)  # (B, N, k, h * units)

        # multi-head layer (B, N, h * units)
        value = tf.add(value, position)
        multi_heads = tf.multiply(value, attention)
        multi_heads = tf.math.reduce_sum(multi_heads, axis=-2)

        # linear projection
        output = self.linear_out(multi_heads)  # (B, N, units)

        # shortcut
        if self.add_shortcut:
            output = tf.add(output, x)  # (B, N, units)

        if self.normalization:
            output = self.norm(output)  # (B, N, units)

        if self.return_heads:
            return output, multi_heads
        else:
            return output

    def get_config(self):
        config = super(MultiAttentionLayer, self).get_config()
        config.update({
            'units': self.units,
            'latent_units': self.latent_units,
            'k': self.k,
            'nb_head': self.nb_head,
            'add_shortcut': self.add_shortcut,
            'normalization': self.normalization,
            'return_heads': self.return_heads})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BlockMultiAttention(Layer):

    def __init__(self,
                 l_units,
                 k,
                 nb_head,
                 add_shortcut=False,
                 normalization=False,
                 return_heads=False,
                 **kwargs):
        super(BlockMultiAttention, self).__init__(**kwargs)

        # initialize parameter
        self.l_units = l_units
        self.k = k
        self.h = nb_head
        self.add_shortcut = add_shortcut
        self.normalization = normalization
        self.return_heads = return_heads

        # define layers
        self.layers = []
        for units in self.l_units:
            layer = MultiAttentionLayer(
                units=units,
                latent_units=int(units / 2),
                k=self.k,
                nb_head=self.h,
                add_shortcut=self.add_shortcut,
                normalization=self.normalization,
                return_heads=False)
            self.layers.append(layer)
        self.layers[-1].return_heads = self.return_heads

    def call(self, inputs, *args, **kwargs):
        # get inputs
        x = inputs

        # compute layers
        for layer in self.layers:
            x = layer(x)

        return x

    def get_config(self):
        config = super(BlockMultiAttention, self).get_config()
        config.update({
            'l_units': self.l_units,
            'k': self.k,
            'nb_head': self.nb_head,
            'add_shortcut': self.add_shortcut,
            'normalization': self.normalization,
            'return_heads': self.return_heads})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def farthest_point_sampling(inputs, downsampling_factor, return_indices=False):
    # inputs (B, N, C)

    # get shape
    batch_size = tf.shape(inputs)[0]
    length = tf.shape(inputs)[1]
    length_s = tf.cast(tf.round(length / downsampling_factor), dtype=tf.int32)

    # initialize tensors
    distances = tf.ones((batch_size, length))
    distances = tf.math.scalar_mul(1e10, distances)  # (B, N)
    farthest_indices = tf.random.uniform(
        shape=(batch_size,),
        minval=0,
        maxval=length,
        seed=13,
        dtype=tf.int32)  # (B,)
    farthest_indices = tf.cast(farthest_indices, dtype=tf.int64)

    # loop over farthest points
    centroids_indices = tf.TensorArray(
        dtype=tf.int64,
        size=length_s)  # [(1, N/s)]
    for i in tf.range(length_s):
        centroids_indices = centroids_indices.write(i, farthest_indices)
        x = tf.gather(
            inputs,
            indices=farthest_indices,
            axis=1,
            batch_dims=1)  # (B, C)
        x = tf.expand_dims(x, axis=-2)  # (B, 1, C)
        x = tf.tile(x, multiples=(1, length, 1))  # (B, N, C)
        x = tf.subtract(inputs, x) ** 2  # (B, N, C)
        x = tf.reduce_sum(x, axis=-1)  # (B, N)
        distances = tf.math.minimum(distances, x)  # (B, N)
        farthest_indices = tf.math.argmax(distances, axis=-1)  # (B,)
    centroids_indices = centroids_indices.stack()  # (N/s, B)
    centroids_indices = tf.transpose(centroids_indices)  # (B, N/s)
    centroids = tf.gather(
        params=inputs,
        indices=centroids_indices,
        axis=1,
        batch_dims=1)  # (B, N/s, C)

    if return_indices:
        return centroids, centroids_indices
    else:
        return centroids


class GeometricAffine(Layer):

    def __init__(self,
                 **kwargs):
        super(GeometricAffine, self).__init__(**kwargs)
        # define layers
        self.maxpool = GlobalMaxPool2D(
            keepdims=True)

        # parameters to build
        self.k = None
        self.c = None
        self.a = None
        self.b = None

    def build(self, input_shape):
        # get number of features and centroids
        self.k = input_shape[1][2]
        self.c = input_shape[0][-1]

        # define affine parameters
        self.a = self.add_weight(
            shape=(1, self.c),
            initializer=tf.ones_initializer,
            trainable=True,
            name="a")
        self.b = self.add_weight(
            shape=(1, self.c),
            initializer=tf.zeros_initializer,
            trainable=True,
            name="b")

    def call(self, inputs, *args, **kwargs):
        # get inputs
        centroids, neighbors = inputs  # (B, N, C), (B, N, k, C)
        centroids = tf.expand_dims(centroids, axis=2)  # (B, N, 1, C)
        centroids = tf.tile(centroids, multiples=(1, 1, self.k, 1))

        # get number of centroids
        nb_centroids = tf.shape(centroids)[1]

        # compute feature deviation (B,)
        sigma = tf.subtract(neighbors, centroids) ** 2  # (B, N, k, C)
        sigma = self.maxpool(sigma)  # (B, 1, 1, C)
        scalar = tf.cast(1/(self.k * nb_centroids * self.c), dtype=tf.float32)
        sigma = tf.scalar_mul(scalar, sigma)  # (B, 1, 1, C)
        sigma = tf.math.reduce_std(sigma, axis=-1, keepdims=True)
        sigma = tf.add(sigma, tf.keras.backend.epsilon())  # (B, 1, 1, 1)
        sigma = 1 / sigma

        # compute edges  (B, N, k, C)
        output = tf.subtract(neighbors, centroids)
        output = tf.multiply(sigma, output)
        output = tf.matmul(output, tf.transpose(self.a)) + self.b

        return output

    def get_config(self):
        config = super(GeometricAffine, self).get_config()

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResidualPoint(Layer):

    def __init__(self,
                 units,
                 **kwargs):
        super(ResidualPoint, self).__init__(**kwargs)

        # initialize parameter
        self.units = units

        # define layers
        self.mlp_0 = CustomConv(
            filters=self.units,
            kernel_size=(1, 1),
            normalization=True,
            activation="relu")
        self.mlp_1 = CustomConv(
            filters=self.units,
            kernel_size=(1, 1),
            normalization=True,
            activation=None)
        self.act = Activation(
            activation="relu")

    def call(self, inputs, *args, **kwargs):  # (B, N, k, units)
        # compute MLPs
        x = self.mlp_0(inputs)  # (B, N, k, units)
        x = self.mlp_1(x)  # (B, N, k, units)

        # residual connection
        x = tf.add(x, inputs)  # (B, N, k, units)

        # final activation
        output = self.act(x)  # (B, N, k, units)

        return output

    def get_config(self):
        config = super(ResidualPoint, self).get_config()
        config.update({
            'units': self.units})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BlockResidualPoint(Layer):

    def __init__(self,
                 l_units,
                 **kwargs):
        super(BlockResidualPoint, self).__init__(**kwargs)

        # initialize parameter
        self.l_units = l_units

        # define layers
        self.layers = []
        for units in self.l_units:
            layer = ResidualPoint(
                units=units)
            self.layers.append(layer)

    def call(self, inputs, *args, **kwargs):
        # get inputs
        x = inputs

        # compute layers
        for layer in self.layers:
            x = layer(x)

        return x

    def get_config(self):
        config = super(BlockResidualPoint, self).get_config()
        config.update({
            'l_units': self.l_units})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BlockPointMLP(Layer):

    def __init__(self,
                 downsampling_factor,
                 units,
                 k,
                 return_centroids_indices=False,
                 **kwargs):
        super(BlockPointMLP, self).__init__(**kwargs)

        # initialize parameter
        self.downsampling_factor = downsampling_factor
        self.units = units
        self.k = k
        self.return_centroids_indices = return_centroids_indices

        # define layers
        self.embedding = CustomDense(
            units=self.units,
            normalization=True,
            activation="relu")
        self.edges_extractor = EdgeFeature(
            k=self.k,
            return_edge_only=True)
        self.block_pre = BlockResidualPoint(
            l_units=(self.units, self.units, self.units))
        self.block_post = BlockResidualPoint(
            l_units=(self.units, self.units, self.units))

    def call(self, inputs, *args, **kwargs):  # (B, N, C)
        # embed inputs
        x = self.embedding(inputs)  # (B, N, units)

        # sample farthest points
        centroids, centroids_indices = farthest_point_sampling(
            inputs=x,
            downsampling_factor=self.downsampling_factor,
            return_indices=True)  # (B, N/s, units), (B, N/s)

        # extract edges features (B, N, k, units)
        x_edge = self.edges_extractor(x)

        # get centroids' neighbors
        x_neighbors = tf.gather(
            x_edge,
            indices=centroids_indices,
            axis=1,
            batch_dims=1)  # (B, N/s, k, units)

        # residual point blocks pre-aggregation (B, N/s, k, units)
        x_neighbors = self.block_pre(x_neighbors)

        # aggregation (B, N/s, 1, units)
        x_neighbors = tf.reduce_max(x_neighbors, axis=2, keepdims=True)

        # residual point blocks post-aggregation (B, N/s, units)
        x_neighbors = self.block_post(x_neighbors)
        x_neighbors = tf.squeeze(x_neighbors, axis=2)

        if self.return_centroids_indices:
            return x_neighbors, centroids_indices
        else:
            return x_neighbors

    def get_config(self):
        config = super(BlockPointMLP, self).get_config()
        config.update({
            'downsampling_factor': self.downsampling_factor,
            'units': self.units,
            'k': self.k,
            'return_centroids_indices': self.return_centroids_indices})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
