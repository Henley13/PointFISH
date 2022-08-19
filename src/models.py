# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Point cloud models.
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.layers import Concatenate

from utils_model import CustomConv
from utils_model import BlockConv
from utils_model import CustomDense
from utils_model import BlockDense
from utils_model import TNet
from utils_model import TNetEdge
from utils_model import get_additional_features
from utils_model import EdgeFeature
from utils_model import EdgeConv
from utils_model import BlockEdge
from utils_model import MultiAttentionLayer
from utils_model import BlockMultiAttention
from utils_model import BlockPointMLP
from utils_model import GeometricAffine

from utils_train import LRSchedule


# ###  Compilation ###

def compile_model(model):
    # loss
    loss = tf.keras.losses.BinaryCrossentropy()

    # metric
    accuracy = tf.metrics.BinaryAccuracy()

    # optimizer
    lr_schedule = LRSchedule(
        decay_steps=20000,
        name="lr_schedule")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        name="Adam")

    # compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=accuracy)

    return model


def initialize_model(
        base_model,
        nb_inputs,
        add_cluster,
        add_morphology,
        add_distance,
        inputs_alignment,
        features_alignment,
        filters_pre,
        filters_post,
        k,
        nb_head,
        nb_classes,
        dropout_rate,
        embedding_regularization,
        name):
    # create model
    if base_model == "PointNet":
        model = build_pointnet_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=False,
            filters_pre=filters_pre,
            filters_post=filters_post,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "MSPointNet":
        model = build_pointnet_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=True,
            filters_pre=filters_pre,
            filters_post=filters_post,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "DGCNN":
        model = build_dgcnn_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=False,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            add_strides=False,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "DGCNNStrides":
        model = build_dgcnn_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=False,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            add_strides=True,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "MSDGCNN":
        model = build_dgcnn_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=True,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            add_strides=False,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "MSDGCNNStrides":
        model = build_dgcnn_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=True,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            add_strides=True,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "PointTransformer":
        model = build_point_transformer_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=False,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "MSPointTransformer":
        model = build_point_transformer_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=True,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization,
            model_name=name)
    elif base_model == "PointMLP":
        model = build_pointmlp_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            model_name=name)
    elif base_model == "PointFISH":
        model = build_pointfish_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=False,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            model_name=name)
    elif base_model == "MSPointFISH":
        model = build_pointfish_model(
            nb_inputs=nb_inputs,
            add_cluster=add_cluster,
            add_morphology=add_morphology,
            add_distance=add_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            multiscale=True,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            model_name=name)
    else:
        valid_model_name = [
            "PointNet", "MSPointNet",
            "DGCNN", "DGCNNStrides", "MSDGCNN", "MSDGCNNStrides",
            "PointTransformer", "MSPointTransformer", "PointMLP",
            "PointFISH", "MSPointFISH"]
        raise ValueError("Model is not recognized. Please choose among:"
                         .format(valid_model_name))

    return model


def get_trainable_variables(model):
    trainable_variables = 0
    for x in model.trainable_variables:
        x = np.array(x)
        trainable_variables += x.size

    return trainable_variables


def get_inputs(inputs, add_cluster, add_morphology, add_distance):
    # check every combinations
    if add_cluster and add_morphology and add_distance:
        (input_coordinate,
         input_cluster, input_morphology, input_distance) = inputs
        extra_features = [input_cluster, input_morphology, input_distance]
        extra_features = tf.concat(extra_features, axis=-1)
    elif not add_cluster and add_morphology and add_distance:
        input_coordinate, input_morphology, input_distance = inputs
        extra_features = [input_morphology, input_distance]
        extra_features = tf.concat(extra_features, axis=-1)
    elif add_cluster and not add_morphology and add_distance:
        input_coordinate, input_cluster, input_distance = inputs
        extra_features = [input_cluster, input_distance]
        extra_features = tf.concat(extra_features, axis=-1)
    elif add_cluster and add_morphology and not add_distance:
        input_coordinate, input_cluster, input_morphology = inputs
        extra_features = [input_cluster, input_morphology]
        extra_features = tf.concat(extra_features, axis=-1)
    elif not add_cluster and not add_morphology and add_distance:
        input_coordinate, input_distance = inputs
        extra_features = input_distance
    elif add_cluster and not add_morphology and not add_distance:
        input_coordinate, input_cluster = inputs
        extra_features = input_cluster
    elif not add_cluster and add_morphology and not add_distance:
        input_coordinate, input_morphology = inputs
        extra_features = input_morphology
    else:
        input_coordinate = inputs
        extra_features = None

    return input_coordinate, extra_features


# ###  PointNet (classification) ###

def get_pointnet_vanilla_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        filters=(64, 64, 64, 128, 1024, 512, 256),
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True):
    # align inputs
    if inputs_alignment:
        x = tf.expand_dims(
            input_coordinate,
            axis=2)  # (B, N, 1, C)
        input_transformer = TNet(
            name="input_transformer")(x)  # (B, C, C)
        input_transformed = tf.matmul(
            input_coordinate,
            input_transformer,
            name="input_transformed")  # (B, N, C)
        input_transformed = tf.expand_dims(
            input_transformed,
            axis=2)  # (B, N, 1, C)
    else:
        input_transformed = tf.expand_dims(
            input_coordinate,
            axis=2)  # (B, N, 1, C)

    # MLP block
    block_0 = BlockConv(
        l_filters=[filters[0], filters[1]],
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_0")(input_transformed)  # (B, N, 1, 64)

    # align features
    feature_transformer = TNet(
        add_regularization=embedding_regularization,
        name="feature_transformer")(block_0)  # (B, 64, 64)
    feature_transformed = tf.matmul(
        tf.squeeze(block_0, axis=2),
        feature_transformer,
        name="feature_transformed")  # (B, N, 64)
    feature_transformed = tf.expand_dims(
        feature_transformed,
        axis=2)  # (B, N, 1, 64)

    # MLP block
    block_1 = BlockConv(
        l_filters=[filters[2], filters[3]],
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(feature_transformed)  # (B, N, 1, 128)

    # additional features
    block_1 = get_additional_features(
        x=block_1,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)

    # MLP block
    block_2 = BlockConv(
        l_filters=[filters[4]],
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_2")(block_1)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_2)  # (B, 1024)

    # MLP block
    block_3 = BlockDense(
        l_units=[filters[5], filters[6]],
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_3")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_3)  # (B, nb_classes)

    return label


def get_pointnet_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True):
    # align inputs
    if inputs_alignment:
        x = tf.expand_dims(
            input_coordinate,
            axis=2)  # (B, N, 1, C)
        input_transformer = TNet(
            name="input_transformer")(x)  # (B, C, C)
        input_transformed = tf.matmul(
            input_coordinate,
            input_transformer,
            name="input_transformed")  # (B, N, C)
        input_transformed = tf.expand_dims(
            input_transformed,
            axis=2)  # (B, N, 1, C)
    else:
        input_transformed = tf.expand_dims(
            input_coordinate,
            axis=2)  # (B, N, 1, C)

    # MLP block
    block_0 = BlockConv(
        l_filters=[64, 64, 64, 64],
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_0")(input_transformed)  # (B, N, 1, 64)

    # align features
    if features_alignment:
        feature_transformer = TNet(
            add_regularization=embedding_regularization,
            name="feature_transformer")(block_0)  # (B, 64, 64)
        feature_transformed = tf.matmul(
            tf.squeeze(block_0, axis=2),
            feature_transformer,
            name="feature_transformed")  # (B, N, 64)
        feature_transformed = tf.expand_dims(
            feature_transformed,
            axis=2)  # (B, N, 1, 64)
    else:
        feature_transformed = block_0

    # additional features
    block_0 = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(block_0)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def get_multiscale_pointnet_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True):
    # align inputs
    if inputs_alignment:
        x = tf.expand_dims(
            input_coordinate,
            axis=2)  # (B, N, 1, C)
        input_transformer = TNet(
            name="input_transformer")(x)  # (B, C, C)
        input_transformed = tf.matmul(
            input_coordinate,
            input_transformer,
            name="input_transformed")  # (B, N, C)
        input_transformed = tf.expand_dims(
            input_transformed,
            axis=2)  # (B, N, 1, C)
    else:
        input_transformed = tf.expand_dims(
            input_coordinate,
            axis=2)  # (B, N, 1, C)

    # MLP block
    mlp_0 = CustomConv(
        filters=64,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="mlp_0")(input_transformed)  # (B, N, 1, 64)
    mlp_1 = CustomConv(
        filters=64,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="mlp_1")(mlp_0)  # (B, N, 1, 64)
    mlp_2 = CustomConv(
        filters=64,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="mlp_2")(mlp_1)  # (B, N, 1, 64)
    mlp_3 = CustomConv(
        filters=64,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="mlp_3")(mlp_2)  # (B, N, 1, 64)
    residual_layers = [mlp_0, mlp_1, mlp_2, mlp_3]

    # align features
    if features_alignment:
        feature_transformer = TNet(
            add_regularization=embedding_regularization,
            name="feature_transformer")(mlp_3)  # (B, 64, 64)
        feature_transformed = tf.matmul(
            tf.squeeze(mlp_3, axis=2),
            feature_transformer,
            name="feature_transformed")  # (B, N, 64)
        feature_transformed = tf.expand_dims(
            feature_transformed,
            axis=2)  # (B, N, 1, 64)
        residual_layers.append(feature_transformed)
    else:
        feature_transformed = mlp_3

    # additional features
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)

    # shortcut connections
    shortcut = Concatenate(
        axis=-1)([mlp_0, mlp_1, mlp_2, feature_transformed])  # (B, N, 1, 256)

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(shortcut)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def build_pointnet_model(
        nb_inputs=3,
        add_cluster=False,
        add_morphology=False,
        add_distance=False,
        inputs_alignment=True,
        features_alignment=True,
        multiscale=False,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True,
        model_name="PointNet"):
    # define input array
    input_coordinate = Input(
        shape=(None, nb_inputs),
        dtype="float32",
        name="input_coordinate")
    inputs = [input_coordinate]
    input_cluster = None
    if add_cluster:
        input_cluster = Input(
            shape=(None, 1),
            dtype="float32",
            name="input_cluster")
        inputs.append(input_cluster)
    input_morphology = None
    if add_morphology:
        input_morphology = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_morphology")
        inputs.append(input_morphology)
    input_distance = None
    if add_distance:
        input_distance = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_distance")
        inputs.append(input_distance)

    # define output
    if multiscale:
        output = get_multiscale_pointnet_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization)
    else:
        output = get_pointnet_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization)

    # define model
    model = Model(
        inputs,
        output,
        name=model_name)

    return model


# ###  DGCNN (classification) ###

def get_dgcnn_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        add_strides=False,
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True):
    # align inputs
    if inputs_alignment:
        input_transformer = TNetEdge(
            k=k,
            name="input_transformer")(input_coordinate)  # (B, C, C)
        input_transformed = tf.matmul(
            input_coordinate,
            input_transformer,
            name="input_transformed")  # (B, N, C)
    else:
        input_transformed = input_coordinate

    # edge convolutions
    block_0 = BlockEdge(
        k,
        l_filters=[64, 64, 64, 64],
        activation="relu",
        add_strides=add_strides,
        normalization=True,
        name="block_0")(input_transformed)  # (B, N, 64)

    # align features
    if features_alignment:
        feature_transformer = TNetEdge(
            k=k,
            add_regularization=embedding_regularization,
            name="feature_transformer")(block_0)  # (B, 64, 64)
        feature_transformed = tf.matmul(
            block_0,
            feature_transformer,
            name="feature_transformed")  # (B, N, 64)
        feature_transformed = tf.expand_dims(
            feature_transformed,
            axis=2)  # (B, N, 1, 64)
    else:
        feature_transformed = tf.expand_dims(
            block_0,
            axis=2)  # (B, N, 1, 64)

    # additional features
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(feature_transformed)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def get_multiscale_dgcnn_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        add_strides=False,
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True):
    # align inputs
    if inputs_alignment:
        input_transformer = TNetEdge(
            k=k,
            name="input_transformer")(input_coordinate)  # (B, C, C)
        input_transformed = tf.matmul(
            input_coordinate,
            input_transformer,
            name="input_transformed")  # (B, N, C)
    else:
        input_transformed = input_coordinate

    # edge convolutions
    edge_conv_0 = EdgeConv(
        k=k,
        filters=64,
        activation="relu",
        add_strides=add_strides,
        normalization=True,
        name="edge_conv_0")(input_transformed)  # (B, N, 64)
    edge_conv_1 = EdgeConv(
        k=k,
        filters=64,
        activation="relu",
        add_strides=add_strides,
        normalization=True,
        name="edge_conv_1")(edge_conv_0)  # (B, N, 64)
    edge_conv_2 = EdgeConv(
        k=k,
        filters=64,
        activation="relu",
        add_strides=add_strides,
        normalization=True,
        name="edge_conv_2")(edge_conv_1)  # (B, N, 64)
    edge_conv_3 = EdgeConv(
        k=k,
        filters=64,
        activation="relu",
        add_strides=add_strides,
        normalization=True,
        name="edge_conv_3")(edge_conv_2)  # (B, N, 64)

    # align features
    if features_alignment:
        feature_transformer = TNetEdge(
            k=k,
            add_regularization=embedding_regularization,
            name="feature_transformer")(edge_conv_3)  # (B, 64, 64)
        feature_transformed = tf.matmul(
            edge_conv_3,
            feature_transformer,
            name="feature_transformed")  # (B, N, 64)
        feature_transformed = tf.expand_dims(
            feature_transformed,
            axis=2)  # (B, N, 1, 64)
    else:
        feature_transformed = tf.expand_dims(
            edge_conv_3,
            axis=2)  # (B, N, 1, 64)

    # additional features
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)

    # shortcut connections (B, N, 1, 256)
    shortcut = Concatenate(
        axis=-1)([tf.expand_dims(edge_conv_0, axis=2),
                  tf.expand_dims(edge_conv_1, axis=2),
                  tf.expand_dims(edge_conv_2, axis=2),
                  feature_transformed])

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(shortcut)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def build_dgcnn_model(
        nb_inputs=3,
        add_cluster=False,
        add_morphology=False,
        add_distance=False,
        inputs_alignment=True,
        features_alignment=True,
        multiscale=False,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        add_strides=False,
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True,
        model_name="DGCNN"):
    # define input array
    input_coordinate = Input(
        shape=(None, nb_inputs),
        dtype="float32",
        name="input_coordinate")
    inputs = [input_coordinate]
    input_cluster = None
    if add_cluster:
        input_cluster = Input(
            shape=(None, 1),
            dtype="float32",
            name="input_cluster")
        inputs.append(input_cluster)
    input_morphology = None
    if add_morphology:
        input_morphology = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_morphology")
        inputs.append(input_morphology)
    input_distance = None
    if add_distance:
        input_distance = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_distance")
        inputs.append(input_distance)

    # define output
    if multiscale:
        output = get_multiscale_dgcnn_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            add_strides=add_strides,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization)
    else:
        output = get_dgcnn_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            add_strides=add_strides,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization)

    # define model
    model = Model(
        inputs,
        output,
        name=model_name)

    return model


# ### Point Transformer (classification) ###

def get_point_transformer_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_head=3,
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True):
    # align inputs
    if inputs_alignment:
        input_transformer = TNetEdge(
            k=k,
            name="input_transformer")(input_coordinate)  # (B, C, C)
        input_transformed = tf.matmul(
            input_coordinate,
            input_transformer,
            name="input_transformed")  # (B, N, C)
    else:
        input_transformed = input_coordinate  # (B, N, C)

    # transformer blocks
    block_0, _ = BlockMultiAttention(
        l_units=(16, 16, 16, 16),
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=True,
        name="block_0")(input_transformed)  # (B, N, 16)

    # align features
    if features_alignment:
        feature_transformer = TNetEdge(
            k=k,
            add_regularization=embedding_regularization,
            name="feature_transformer")(block_0)  # (B, 16, 16)
        feature_transformed = tf.matmul(
            block_0,
            feature_transformer,
            name="feature_transformed")  # (B, N, 16)
        feature_transformed = tf.expand_dims(
            feature_transformed,
            axis=2)  # (B, N, 1, 16)
    else:
        feature_transformed = tf.expand_dims(
            block_0,
            axis=2)  # (B, N, 1, 16)

    # additional features
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)  # (B, N, 1, 32)

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(feature_transformed)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def get_multiscale_point_transformer_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_head=3,
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True):
    # align inputs
    if inputs_alignment:
        input_transformer = TNetEdge(
            k=k,
            add_regularization=embedding_regularization,
            name="input_transformer")(input_coordinate)  # (B, C, C)
        input_transformed = tf.matmul(
            input_coordinate,
            input_transformer,
            name="input_transformed")  # (B, N, C)
    else:
        input_transformed = input_coordinate  # (B, N, C)

    # transformer blocks
    multi_attention_0 = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        name="multi_attention_0")(input_transformed)  # (B, N, 16)
    multi_attention_1 = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=False,
        name="multi_attention_1")(multi_attention_0)  # (B, N, 16)
    multi_attention_2 = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=False,
        name="multi_attention_2")(multi_attention_1)  # (B, N, 16)
    multi_attention_3, _ = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=True,
        name="multi_attention_3")(multi_attention_2)  # (B, N, 16)

    # align features
    if features_alignment:
        feature_transformer = TNetEdge(
            k=k,
            add_regularization=embedding_regularization,
            name="feature_transformer")(multi_attention_3)  # (B, 16, 16)
        feature_transformed = tf.matmul(
            multi_attention_3,
            feature_transformer,
            name="feature_transformed")  # (B, N, 16)
        feature_transformed = tf.expand_dims(
            feature_transformed,
            axis=2)  # (B, N, 1, 16)
    else:
        feature_transformed = tf.expand_dims(
            multi_attention_3,
            axis=2)  # (B, N, 1, 16)

    # additional features
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)  # (B, N, 1, 32)

    # shortcut connections
    shortcut = Concatenate(
        axis=-1)([tf.expand_dims(multi_attention_0, axis=2),
                  tf.expand_dims(multi_attention_1, axis=2),
                  tf.expand_dims(multi_attention_2, axis=2),
                  feature_transformed])  # (B, N, 1, 64)

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(shortcut)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def build_point_transformer_model(
        nb_inputs=3,
        add_cluster=False,
        add_morphology=False,
        add_distance=False,
        inputs_alignment=True,
        features_alignment=True,
        multiscale=False,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_head=3,
        nb_classes=8,
        dropout_rate=0.1,
        embedding_regularization=True,
        model_name="PointTransformer"):
    # define input array
    input_coordinate = Input(
        shape=(None, nb_inputs),
        dtype="float32",
        name="input_coordinate")
    inputs = [input_coordinate]
    input_cluster = None
    if add_cluster:
        input_cluster = Input(
            shape=(None, 1),
            dtype="float32",
            name="input_cluster")
        inputs.append(input_cluster)
    input_morphology = None
    if add_morphology:
        input_morphology = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_morphology")
        inputs.append(input_morphology)
    input_distance = None
    if add_distance:
        input_distance = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_distance")
        inputs.append(input_distance)

    # define output
    if multiscale:
        output = get_multiscale_point_transformer_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization)
    else:
        output = get_point_transformer_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate,
            embedding_regularization=embedding_regularization)

    # define model
    model = Model(
        inputs,
        output,
        name=model_name)

    return model


# ### PointMLP (classification) ###

def get_pointmlp_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_classes=8,
        dropout_rate=0.1):
    # align inputs
    if inputs_alignment:
        neighbors_coordinate = EdgeFeature(
            k=k,
            return_edge_only=True,
            name="input_neighbors")(input_coordinate)  # (B, N, k, C)
        input_transformed = GeometricAffine(
            name="input_transformed")([input_coordinate, neighbors_coordinate])
        input_transformed = tf.reduce_max(
            input_transformed, axis=2)  # (B, N, C)
    else:
        input_transformed = input_coordinate  # (B, N, C)

    # transformer blocks
    block_0 = BlockPointMLP(
        downsampling_factor=1,
        units=64,
        k=k,
        name="block_0")(input_transformed)  # (B, N, 64)
    block_1, centroids_indices_1 = BlockPointMLP(
        downsampling_factor=2,
        units=64,
        k=int(k / 2),
        return_centroids_indices=True,
        name="block_1")(block_0)  # (B, N/2, 64)
    block_2, centroids_indices_2 = BlockPointMLP(
        downsampling_factor=2,
        units=64,
        k=int(k / 4),
        return_centroids_indices=True,
        name="block_2")(block_1)  # (B, N/4, 64)
    block_3 = BlockPointMLP(
        downsampling_factor=1,
        units=64,
        k=int(k / 4),
        name="block_3")(block_2)  # (B, N/4, 64)

    # align features
    if features_alignment:
        neighbors_coordinate = EdgeFeature(
            k=int(k / 4),
            return_edge_only=True,
            name="feature_neighbors")(block_3)  # (B, N/4, k/4, 64)
        feature_transformed = GeometricAffine(
            name="feature_transformed")([block_3, neighbors_coordinate])
        feature_transformed = tf.reduce_max(
            feature_transformed, axis=2, keepdims=True)  # (B, N/4, 1, 64)
    else:
        feature_transformed = tf.expand_dims(
            block_3,
            axis=2)  # (B, N/4, 1, 64)

    # additional features (B, N/4, 1, 80)
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance,
        downsampling_indices=(centroids_indices_1, centroids_indices_2))

    # MLP block
    block_4 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_4")(feature_transformed)  # (B, N/4, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_4)  # (B, 1024)

    # MLP block
    block_5 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_5")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_5)  # (B, nb_classes)

    return label


def build_pointmlp_model(
        nb_inputs=3,
        add_cluster=False,
        add_morphology=False,
        add_distance=False,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_classes=8,
        dropout_rate=0.1,
        model_name="PointMLP"):
    # define input array
    input_coordinate = Input(
        shape=(None, nb_inputs),
        dtype="float32",
        name="input_coordinate")
    inputs = [input_coordinate]
    input_cluster = None
    if add_cluster:
        input_cluster = Input(
            shape=(None, 1),
            dtype="float32",
            name="input_cluster")
        inputs.append(input_cluster)
    input_morphology = None
    if add_morphology:
        input_morphology = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_morphology")
        inputs.append(input_morphology)
    input_distance = None
    if add_distance:
        input_distance = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_distance")
        inputs.append(input_distance)

    # define output
    output = get_pointmlp_model(
        input_coordinate=input_coordinate,
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance,
        inputs_alignment=inputs_alignment,
        features_alignment=features_alignment,
        filters_pre=filters_pre,
        filters_post=filters_post,
        k=k,
        nb_classes=nb_classes,
        dropout_rate=dropout_rate)

    # define model
    model = Model(
        inputs,
        output,
        name=model_name)

    return model


# ### FISH Point (classification) ###

def get_pointfish_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_head=3,
        nb_classes=8,
        dropout_rate=0.1):
    # align inputs
    if inputs_alignment:
        neighbors_coordinate = EdgeFeature(
            k=k,
            return_edge_only=True,
            name="input_neighbors")(input_coordinate)  # (B, N, k, C)
        input_transformed = GeometricAffine(
            name="input_transformed")([input_coordinate, neighbors_coordinate])
        input_transformed = tf.reduce_max(
            input_transformed, axis=2)  # (B, N, C)
    else:
        input_transformed = input_coordinate  # (B, N, C)

    # transformer blocks
    block_0, _ = BlockMultiAttention(
        l_units=(16, 16, 16, 16),
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=True,
        name="block_0")(input_transformed)  # (B, N, 16)

    # align features
    if features_alignment:
        neighbors_coordinate = EdgeFeature(
            k=k,
            return_edge_only=True,
            name="feature_neighbors")(block_0)  # (B, N, k, 16)
        feature_transformed = GeometricAffine(
            name="feature_transformed")([block_0, neighbors_coordinate])
        feature_transformed = tf.reduce_max(
            feature_transformed, axis=2, keepdims=True)  # (B, N, 1, 16)
    else:
        feature_transformed = tf.expand_dims(
            block_0,
            axis=2)  # (B, N, 1, 16)

    # additional features
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)  # (B, N, 1, 32)

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(feature_transformed)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def get_multiscale_pointfish_model(
        input_coordinate,
        input_cluster=None,
        input_morphology=None,
        input_distance=None,
        inputs_alignment=True,
        features_alignment=True,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_head=3,
        nb_classes=8,
        dropout_rate=0.1):
    # align inputs
    if inputs_alignment:
        neighbors_coordinate = EdgeFeature(
            k=k,
            return_edge_only=True,
            name="input_neighbors")(input_coordinate)  # (B, N, k, C)
        input_transformed = GeometricAffine(
            name="input_transformed")([input_coordinate, neighbors_coordinate])
        input_transformed = tf.reduce_max(
            input_transformed, axis=2)  # (B, N, C)
    else:
        input_transformed = input_coordinate  # (B, N, C)

    # transformer blocks
    multi_attention_0 = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        name="multi_attention_0")(input_transformed)  # (B, N, 16)
    multi_attention_1 = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=False,
        name="multi_attention_1")(multi_attention_0)  # (B, N, 16)
    multi_attention_2 = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=False,
        name="multi_attention_2")(multi_attention_1)  # (B, N, 16)
    multi_attention_3, _ = MultiAttentionLayer(
        units=16,
        latent_units=8,
        k=k,
        nb_head=nb_head,
        add_shortcut=True,
        normalization=True,
        return_heads=True,
        name="multi_attention_3")(multi_attention_2)  # (B, N, 16)

    # align features
    if features_alignment:
        neighbors_coordinate = EdgeFeature(
            k=k,
            return_edge_only=True,
            name="feature_neighbors")(multi_attention_3)  # (B, N, k, 16)
        feature_transformed = GeometricAffine(
            name="feature_transformed")([multi_attention_3,
                                         neighbors_coordinate])
        feature_transformed = tf.reduce_max(
            feature_transformed, axis=2, keepdims=True)  # (B, N, 1, 16)
    else:
        feature_transformed = tf.expand_dims(
            multi_attention_3,
            axis=2)  # (B, N, 1, 16)

    # additional features
    feature_transformed = get_additional_features(
        x=feature_transformed,
        filters=(16, 16),
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance)  # (B, N, 1, 32)

    # shortcut connections
    shortcut = Concatenate(
        axis=-1)([tf.expand_dims(multi_attention_0, axis=2),
                  tf.expand_dims(multi_attention_1, axis=2),
                  tf.expand_dims(multi_attention_2, axis=2),
                  feature_transformed])  # (B, N, 1, 64)

    # MLP block
    block_1 = BlockConv(
        l_filters=filters_pre,
        kernel_size=(1, 1),
        normalization=True,
        activation="relu",
        name="block_1")(shortcut)  # (B, N, 1, 1024)

    # global feature
    global_feature = GlobalMaxPool2D(
        name="global_feature")(block_1)  # (B, 1024)

    # MLP block
    block_2 = BlockDense(
        l_units=filters_post,
        normalization=True,
        activation="relu",
        dropout_rate=dropout_rate,
        name="block_2")(global_feature)  # (B, 256)

    # output
    label = CustomDense(
        units=nb_classes,
        activation="softmax",
        name="label")(block_2)  # (B, nb_classes)

    return label


def build_pointfish_model(
        nb_inputs=3,
        add_cluster=False,
        add_morphology=False,
        add_distance=False,
        inputs_alignment=True,
        features_alignment=True,
        multiscale=False,
        filters_pre=(128, 256, 512, 1024),
        filters_post=(512, 256),
        k=20,
        nb_head=3,
        nb_classes=8,
        dropout_rate=0.1,
        model_name="PointFISH"):
    # define input array
    input_coordinate = Input(
        shape=(None, nb_inputs),
        dtype="float32",
        name="input_coordinate")
    inputs = [input_coordinate]
    input_cluster = None
    if add_cluster:
        input_cluster = Input(
            shape=(None, 1),
            dtype="float32",
            name="input_cluster")
        inputs.append(input_cluster)
    input_morphology = None
    if add_morphology:
        input_morphology = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_morphology")
        inputs.append(input_morphology)
    input_distance = None
    if add_distance:
        input_distance = Input(
            shape=(None, 2),
            dtype="float32",
            name="input_distance")
        inputs.append(input_distance)

    # define output
    if multiscale:
        output = get_multiscale_pointfish_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate)
    else:
        output = get_pointfish_model(
            input_coordinate=input_coordinate,
            input_cluster=input_cluster,
            input_morphology=input_morphology,
            input_distance=input_distance,
            inputs_alignment=inputs_alignment,
            features_alignment=features_alignment,
            filters_pre=filters_pre,
            filters_post=filters_post,
            k=k,
            nb_head=nb_head,
            nb_classes=nb_classes,
            dropout_rate=dropout_rate)

    # define model
    model = Model(
        inputs,
        output,
        name=model_name)

    return model
