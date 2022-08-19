# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Train and evaluate point cloud classification.
"""

import os
import argparse
import time
import joblib

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

import bigfish.stack as stack

from sklearn.preprocessing import StandardScaler
from umap import UMAP
from tensorflow.python.keras.engine.training import Model

from utils import check_directories
from utils import initialize_script
from utils import end_script

from preprocessing import get_real_binary_mask
from preprocessing import extract_internal_boundary_coord
from preprocessing import build_dataset
from preprocessing import parse_example_clf
from preprocessing import build_features

from models import initialize_model
from models import get_trainable_variables

from utils_train import train_step_clf
from utils_train import val_step_clf
from utils_train import define_callbacks
from utils_train import LRSchedule
from utils_train import plot_confusion_matrix
from utils_train import plot_embedding_real
from utils_train import plot_embedding_probability_real
from utils_train import plot_embedding_genes_real
from utils_train import plot_embedding_genes_by_pattern_real
from utils_train import plot_dendrograms
from utils_train import predict_default_random
from utils_train import evaluation_clf
from utils_train import train_evaluate_ml_models
from utils_train import extract_best_ml_models
from utils_train import plot_boxplot_balanced_accuracy
from utils_train import plot_boxplot_f1
from utils_train import plot_boxplot_precision
from utils_train import plot_boxplot_recall
from utils_train import plot_boxplot_auc

# parameters
nb_classes = 8
batch_size = 32
embedding_regularization = True
dropout_rate = 0.1
nb_epochs = 150

# Your CPU supports instructions that this TensorFlow binary was not compiled
# to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"

# random seed
random_seed = None
# random.seed(random_seed)
# np.random.seed(random_seed)
# tf.random.set_seed(random_seed)


if __name__ == "__main__":
    print()

    #######################
    # ### LAUNCH SCRIPT ###
    #######################

    # allocate one GPU device and limit its memory
    physical_gpu = tf.config.experimental.list_physical_devices('GPU')
    print("Number of physical GPUs: {0}".format(len(physical_gpu)))
    for gpu in physical_gpu:
        print("\r", gpu)

    # get number of CPUs
    nb_cpu = os.cpu_count()
    print("Number of CPUs: {0}".format(nb_cpu))
    print()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory",
                        help="Path of the data directory.",
                        type=str)
    parser.add_argument("output_directory",
                        help="Path of the output directory.",
                        type=str)
    parser.add_argument("add_cluster",
                        help="Use cluster features.",
                        type=int)
    parser.add_argument("add_morphology",
                        help="Use morphological coordinates.",
                        type=int)
    parser.add_argument("add_distance",
                        help="Use distance features.",
                        type=int)
    parser.add_argument("input_dimension",
                        help="Input dimension.",
                        type=int)
    parser.add_argument("base_model_name",
                        help="Model used.",
                        type=str)
    parser.add_argument("inputs_alignment",
                        help="Align inputs.",
                        type=int)
    parser.add_argument("features_alignment",
                        help="Align features.",
                        type=int)
    parser.add_argument("k",
                        help="Number of neighbors to consider.",
                        type=int)
    parser.add_argument("nb_head",
                        help="Number of attention layers to compute.",
                        type=int)
    parser.add_argument("latent_dimension",
                        help="Embedding dimension.",
                        type=int)
    parser.add_argument("array_id",
                        help="Id of the slurm array.",
                        type=int)

    # parse arguments
    args = parser.parse_args()
    data_directory = args.data_directory
    output_directory = args.output_directory
    add_cluster = bool(args.add_cluster)
    add_morphology = bool(args.add_morphology)
    add_distance = bool(args.add_distance)
    nb_inputs = int(args.input_dimension)
    base_model_name = args.base_model_name
    inputs_alignment = bool(args.inputs_alignment)
    features_alignment = bool(args.features_alignment)
    k = int(args.k)
    nb_head = int(args.nb_head)
    latent_dimension = int(args.latent_dimension)
    array_id = int(args.array_id)

    # initialize parameters
    model_name = base_model_name
    training_directory = os.path.join(output_directory, "models_training")
    real_directory = os.path.join(data_directory, "reals")
    if add_cluster:
        model_name += "Cluster"
    if add_morphology:
        tfrecords_directory = os.path.join(
            data_directory, "data_clf", "tfrecords_morphology_clf")
        model_name += "Morphology"
        low_length = 50 + 400
        high_length = 950 + 400
    else:
        tfrecords_directory = os.path.join(
            data_directory, "data_clf", "tfrecords_clf")
        low_length = 50
        high_length = 950
    if add_distance:
        model_name += "Distance"
    if nb_inputs == 3:
        model_name = "3D" + model_name
    else:
        model_name = "2D" + model_name
    if latent_dimension == 256:
        filters_pre = (128, 256, 512, 1024)
        filters_post = (512, 256)
    elif latent_dimension == 128:
        filters_pre = (128, 256, 512)
        filters_post = (256, 128)
    elif latent_dimension == 64:
        filters_pre = (128, 256)
        filters_post = (128, 64)
    elif latent_dimension == 32:
        filters_pre = (128,)
        filters_post = (64, 32)
    else:
        raise ValueError("'latent_dimension' should be among [256, 128, 64, "
                         "32], not {0}.".format(latent_dimension))

    # check directories exists
    check_directories([data_directory,
                       tfrecords_directory, real_directory,
                       output_directory, training_directory])

    # initialize script
    _, log_directory = initialize_script(
        training_directory, experiment_name=model_name, array_id=array_id)
    plot_directory = os.path.join(log_directory, "plots")
    os.mkdir(plot_directory)
    print("Add cluster: {0}".format(add_cluster))
    print("Add morphology: {0}".format(add_morphology))
    print("Add distance: {0}".format(add_distance))
    print("Input dimension: {0}".format(nb_inputs))
    print("Number of classes: {0}".format(nb_classes))
    print("Model: {0}".format(model_name))
    print("Inputs alignment: {0}".format(inputs_alignment))
    print("Features alignment: {0}".format(features_alignment))
    print("Filters pre-pooling: {0}".format(filters_pre))
    print("Filters post-pooling: {0}".format(filters_post))
    if "DGCNN" in base_model_name or "PointTransformer" in base_model_name:
        print("Neighbors: {0}".format(k))
    if "PointTransformer" in base_model_name or "PointFISH" in base_model_name:
        print("Attention heads: {0}".format(nb_head))
    print("Batch size: {0}".format(batch_size))
    print("Maximum number of epochs: {0}".format(nb_epochs), "\n")

    ########################
    # ### INITIALIZATION ###
    ########################

    # parsing functions
    def parsing_fct(x):
        return parse_example_clf(
            x,
            add_label=True,
            add_metadata=False,
            input_dimension=nb_inputs)

    def parse_fct_test(x):
        return parse_example_clf(
            x,
            add_label=True,
            add_metadata=True,
            input_dimension=nb_inputs)

    def parse_fct_real(x):
        return parse_example_clf(
            x,
            add_label=False,
            add_metadata=True,
            input_dimension=nb_inputs)

    # gather paths tfrecords
    path_tfrecords_train = os.path.join(tfrecords_directory, "train")
    path_tfrecords_val = os.path.join(tfrecords_directory, "validation")
    path_tfrecords_test = os.path.join(tfrecords_directory, "test")

    # create datasets
    train_dataset = build_dataset(
        path_tfrecords_directories=path_tfrecords_train,
        function_parse=parsing_fct,
        batch_size=batch_size,
        low_length=low_length,
        high_length=high_length)
    val_dataset = build_dataset(
        path_tfrecords_directories=path_tfrecords_val,
        function_parse=parsing_fct,
        batch_size=batch_size,
        low_length=low_length,
        high_length=high_length)
    test_dataset = build_dataset(
        path_tfrecords_directories=path_tfrecords_test,
        function_parse=parse_fct_test,
        batch_size=1)

    # create model
    model = initialize_model(
        base_model=base_model_name,
        nb_inputs=nb_inputs,
        add_cluster=add_cluster,
        add_morphology=add_morphology,
        add_distance=add_distance,
        inputs_alignment=inputs_alignment,
        features_alignment=features_alignment,
        filters_pre=filters_pre,
        filters_post=filters_post,
        k=k,
        nb_head=nb_head,
        nb_classes=nb_classes,
        dropout_rate=dropout_rate,
        embedding_regularization=embedding_regularization,
        name=model_name)

    # plot model architecture
    path = os.path.join(plot_directory, "{0}.png".format(model_name))
    tf.keras.utils.plot_model(
        model,
        to_file=path,
        show_shapes=True)

    # instantiate loss
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # instantiate optimizer
    lr_schedule = LRSchedule(
        decay_steps=20000,
        name="lr_schedule")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        name="Adam")

    # initialize metrics
    train_f1 = tfa.metrics.F1Score(
        num_classes=nb_classes,
        average="macro",
        threshold=0.5)
    val_f1 = tfa.metrics.F1Score(
        num_classes=nb_classes,
        average="macro",
        threshold=0.5)

    # initialize callbacks
    callbacks = define_callbacks(
        log_directory_experiment=log_directory,
        model=model,
        patience=10)

    ############################
    # ### SIMULATED DATASET  ###
    ############################

    print("--- TRAINING DL --- \n")
    start_time = time.time()

    # loop over epochs
    i = None
    step = 0
    epoch_logs = None
    callbacks.on_train_begin()
    for epoch in range(nb_epochs):

        callbacks.on_epoch_begin(epoch)

        # reset metrics
        train_f1.reset_states()
        val_f1.reset_states()

        # loop over training batches
        train_loss_epoch = 0
        train_regularization_epoch = 0
        for i, (x_train, y_train) in train_dataset.enumerate():

            callbacks.on_train_batch_begin(i)

            # forward pass and backpropagation
            (train_output,
             train_loss_value,
             train_regularization) = train_step_clf(
                inputs=x_train,
                labels=y_train["label"],
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer)

            # update metrics
            train_loss_epoch += train_loss_value
            train_regularization_epoch += train_regularization
            train_f1.update_state(y_train["label"], train_output)
            step += 1

            callbacks.on_train_batch_end(i)

        train_loss_epoch /= tf.cast(i + 1, dtype=tf.float32)
        train_regularization_epoch /= tf.cast(i + 1, dtype=tf.float32)

        # loop over validation batches
        val_loss_epoch = 0
        val_regularization_epoch = 0
        for i, (x_val, y_val) in val_dataset.enumerate():

            callbacks.on_test_batch_begin(i)

            # forward pass
            val_output, val_loss_value, val_regularization = val_step_clf(
                inputs=x_val,
                labels=y_val["label"],
                model=model,
                loss_fn=loss_fn)

            # update metrics
            val_loss_epoch += val_loss_value
            val_regularization_epoch += val_regularization
            val_f1.update_state(y_val["label"], val_output)

            callbacks.on_test_batch_end(i)

        val_loss_epoch /= tf.cast(i + 1, dtype=tf.float32)
        val_regularization_epoch /= tf.cast(i + 1, dtype=tf.float32)

        epoch_logs = {
            "loss": train_loss_epoch,
            "regularization": train_regularization_epoch,
            "f1_score": train_f1.result(),
            "val_loss": val_loss_epoch,
            "val_regularization": val_regularization_epoch,
            "val_f1_score": val_f1.result(),
            "lr": optimizer._decayed_lr(tf.float32)}
        callbacks.on_epoch_end(epoch, epoch_logs)

        if model.stop_training:
            print("Model has been trained for {0} epochs.".format(epoch + 1))
            break

    callbacks.on_train_end(epoch_logs)
    print()

    #  time control
    end_script(start_time)
    start_time = time.time()

    # load model best weights
    model = initialize_model(
        base_model=base_model_name,
        nb_inputs=nb_inputs,
        add_cluster=add_cluster,
        add_morphology=add_morphology,
        add_distance=add_distance,
        inputs_alignment=inputs_alignment,
        features_alignment=features_alignment,
        filters_pre=filters_pre,
        filters_post=filters_post,
        k=k,
        nb_head=nb_head,
        nb_classes=nb_classes,
        dropout_rate=dropout_rate,
        embedding_regularization=embedding_regularization,
        name=model_name)
    path_checkpoint = os.path.join(log_directory, "checkpoint")
    model.load_weights(path_checkpoint)

    print("--- EVALUATION DL --- \n")

    # evaluate training dataset
    train_dataset = build_dataset(
        path_tfrecords_directories=path_tfrecords_train,
        function_parse=parsing_fct,
        batch_size=1)
    train_metrics, _, _, _ = evaluation_clf(
        trained_model=model,
        dataset=train_dataset)

    print("Train results")
    print("\r F1-score: {0:0.3f}".format(train_metrics["f1"]))
    print("\r Precision: {0:0.3f}".format(train_metrics["precision"]))
    print("\r Recall: {0:0.3f}".format(train_metrics["recall"]))
    print()

    # evaluate validation dataset
    val_dataset = build_dataset(
        path_tfrecords_directories=path_tfrecords_val,
        function_parse=parsing_fct,
        batch_size=1)
    val_metrics, _, _, _ = evaluation_clf(
        trained_model=model, dataset=val_dataset)

    print("Validation results")
    print("\r F1-score: {0:0.3f}".format(val_metrics["f1"]))
    print("\r Precision: {0:0.3f}".format(val_metrics["precision"]))
    print("\r Recall: {0:0.3f}".format(val_metrics["recall"]))
    print()

    # evaluate test dataset
    test_dataset = build_dataset(
        path_tfrecords_directories=path_tfrecords_test,
        function_parse=parse_fct_test,
        batch_size=1)
    test_metrics, probabilities, predictions, labels = evaluation_clf(
        trained_model=model,
        dataset=test_dataset)

    print("Test results")
    print("\r F1-score: {0:0.3f}".format(test_metrics["f1"]))
    print("\r Precision: {0:0.3f}".format(test_metrics["precision"]))
    print("\r Recall: {0:0.3f}".format(test_metrics["recall"]))
    print()

    # plot confusion matrices
    path = os.path.join(plot_directory, "confusion_matrix.png")
    plot_confusion_matrix(
        y_true=labels,
        y_pred=predictions,
        path_output=path,
        extension="png")
    path = os.path.join(plot_directory, "confusion_matrix.pdf")
    plot_confusion_matrix(
        y_true=labels,
        y_pred=predictions,
        path_output=path,
        extension="pdf")

    #  time control
    end_script(start_time)
    start_time = time.time()

    ######################
    # ### REAL DATASET ###
    ######################

    print("--- REAL DATASET EMBEDDING --- \n")

    # create model embedding
    inputs_layer = [model.get_layer("input_coordinate").output]
    if add_cluster:
        inputs_layer.append(model.get_layer("input_cluster").output)
    if add_morphology:
        inputs_layer.append(model.get_layer("input_morphology").output)
    if add_distance:
        inputs_layer.append(model.get_layer("input_distance").output)
    if len(inputs_layer) == 1:
        inputs_layer = inputs_layer[0]
    output_embedding = model.get_layer("block_2").output
    model_name_embedding = model_name + "_embedding"
    model_embedding = Model(
        inputs_layer,
        output_embedding,
        name=model_name_embedding)

    # plot model architecture
    path = os.path.join(
        plot_directory, "{0}.png".format(model_name_embedding))
    tf.keras.utils.plot_model(
        model_embedding,
        to_file=path,
        show_shapes=True)

    # read dataframe real
    path = os.path.join(data_directory, "df_real.csv")
    df_real = stack.read_dataframe_from_csv(path)

    # loop over filenames
    embeddings = []
    for i, row in df_real.iterrows():

        # get filename and pattern
        filename = row["cell"]

        # read data
        path = os.path.join(real_directory, "{0}.npz".format(filename))
        data = stack.read_cell_extracted(path)
        rna_coord = data["rna"][:, :3]
        cell_label = data["mask_cyt"]
        nuc_label = data["mask_nuc"]

        # get binary masks
        cell_mask, nuc_mask = get_real_binary_mask(cell_label, nuc_label)

        # bugfix (ugly)
        smoothness = 5
        if i == 6090:
            m = rna_coord[:, 1] == 170
            rna_coord = rna_coord[~m]
        if i == 5803:
            cell_mask = cell_label == 35
            nuc_mask = nuc_label == 35
        if i == 6713:
            smoothness = 6
        if i == 6883:
            cell_mask = cell_label == 9
            nuc_mask = nuc_label == 9

        # get cell and nucleus coordinates (internal boundaries)
        cell_coord = extract_internal_boundary_coord(
            cell_mask,
            smooth_mask=True,
            smoothness=smoothness)
        nuc_coord = extract_internal_boundary_coord(
            nuc_mask,
            smooth_mask=True,
            smoothness=smoothness)

        # build features
        n_coord_cell = 300
        n_coord_nuc = 100
        voxel_size = (300, 103, 103)
        (input_coordinate,
         input_cluster,
         input_morphology,
         input_distance) = build_features(
            rna_coord=rna_coord,
            cell_coord=cell_coord,
            nuc_coord=nuc_coord,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            add_cluster=add_cluster,
            voxel_size=voxel_size,
            add_morphology=add_morphology,
            n_coord_cell=n_coord_cell,
            n_coord_nuc=n_coord_nuc,
            add_distance=add_distance)

        # format features
        features = []
        input_coordinate = input_coordinate[np.newaxis, :, 3 - nb_inputs:]
        features.append(input_coordinate)
        if add_cluster:
            input_cluster = input_cluster[np.newaxis, ...]
            features.append(input_cluster)
        if add_morphology:
            input_morphology = input_morphology[np.newaxis, ...]
            features.append(input_morphology)
        if add_distance:
            input_distance = input_distance[np.newaxis, ...]
            features.append(input_distance)
        if len(features) == 1:
            features = features[0]

        # get results
        embedding = model_embedding.predict_on_batch(features)
        embeddings.append(embedding)

    # format results
    embeddings = np.concatenate(embeddings)

    # save results embeddings
    path = os.path.join(log_directory, "clf_embeddings_real.npy")
    stack.save_array(embeddings, path)

    # normalize embedding
    scaler = StandardScaler()
    normalized_embeddings = embeddings.copy()
    normalized_embeddings = scaler.fit_transform(normalized_embeddings)

    # apply UMAP
    umap = UMAP(
        n_neighbors=50,
        n_components=2,
        metric='euclidean',
        random_state=random_seed)
    embedding_2d_umap = umap.fit_transform(normalized_embeddings)

    # UMAP plots
    plot_directory_embedding = os.path.join(plot_directory, "embedding")
    os.mkdir(plot_directory_embedding)
    path = os.path.join(
        plot_directory_embedding, "umap_real.png")
    plot_embedding_real(
        embedding_2d=embedding_2d_umap,
        df=df_real,
        figsize=(15, 10),
        legend="outside",
        path_output=path,
        extension="png")
    path = os.path.join(
        plot_directory_embedding, "umap_real.pdf")
    plot_embedding_real(
        embedding_2d=embedding_2d_umap,
        df=df_real,
        figsize=(15, 10),
        legend="outside",
        path_output=path,
        extension="pdf")
    path = os.path.join(
        plot_directory_embedding, "umap_real_genes.png")
    plot_embedding_genes_real(
        embedding_2d=embedding_2d_umap,
        df=df_real,
        figsize=(30, 20),
        legend="outside",
        path_output=path,
        extension="png")
    path = os.path.join(
        plot_directory_embedding, "umap_real_genes.pdf")
    plot_embedding_genes_real(
        embedding_2d=embedding_2d_umap,
        df=df_real,
        figsize=(30, 20),
        legend="outside",
        path_output=path,
        extension="pdf")
    patterns = [
        "random",
        "foci", "intranuclear", "nuclear_edge",
        "perinuclear", "cell_edge", "protrusion"]
    for pattern in patterns:
        path = os.path.join(
            plot_directory_embedding,
            "umap_real_genes_{0}.png".format(pattern))
        plot_embedding_genes_by_pattern_real(
            embedding_2d=embedding_2d_umap,
            df=df_real,
            pattern=pattern,
            figsize=(15, 10),
            legend="outside",
            path_output=path,
            extension="png")
        path = os.path.join(
            plot_directory_embedding,
            "umap_real_genes_{0}.pdf".format(pattern))
        plot_embedding_genes_by_pattern_real(
            embedding_2d=embedding_2d_umap,
            df=df_real,
            pattern=pattern,
            figsize=(15, 10),
            legend="outside",
            path_output=path,
            extension="pdf")

    # get hand-crafted features
    features = [
        "nb_foci", "proportion_rna_in_foci",
        "index_foci_mean_distance_cyt", "index_foci_mean_distance_nuc",
        "proportion_rna_in_nuc",
        "index_mean_distance_cyt", "index_mean_distance_nuc",
        "index_rna_opening_30", "index_peripheral_dispersion",
        "index_rna_nuc_edge", "index_rna_nuc_radius_5_10",
        "index_rna_nuc_radius_10_15",
        "index_rna_cyt_radius_0_5", "index_rna_cyt_radius_5_10",
        "index_rna_cyt_radius_10_15"]
    manual_features = df_real.loc[:, features].to_numpy()

    # dendrogram plots
    plot_directory_dendrogram = os.path.join(plot_directory, "dendrogram")
    os.mkdir(plot_directory_dendrogram)
    rand_score_learned, rand_score_manual = plot_dendrograms(
        learned_features=embeddings,
        manual_features=manual_features,
        df=df_real,
        plot_directory=plot_directory_dendrogram)

    print("Adjusted Rand Index: {0:0.3f} (learned) | {1:0.3f} (hand-crafted)"
          .format(rand_score_learned, rand_score_manual), "\n")

    #  time control
    end_script(start_time)
    start_time = time.time()

    print("--- TRAINING ML --- \n")

    # parameters
    pattern_flags = [
        "pattern_foci",
        "pattern_intranuclear", "pattern_nuclear", "pattern_perinuclear",
        "pattern_cell", "pattern_protrusion"]
    patterns = [
        "foci",
        "intranuclear", "nuclear_edge", "perinuclear",
        "cell_edge", "protrusion"]

    print("(f1-score: learned | hand-crafted)", "\n")

    # loop over patterns
    results_embedding = {}
    results_features = {}
    probabilities_rf = []
    probabilities_svc = []
    total_rf_embedding = []
    total_rf_features = []
    total_svc_embedding = []
    total_svc_features = []
    for i, pattern in enumerate(patterns):

        if pattern == "cell_edge":

            # compute probabilities
            probability_rf = np.zeros((len(embeddings), 1))
            probabilities_rf.append(probability_rf)
            probability_svc = np.zeros((len(embeddings), 1))
            probabilities_svc.append(probability_svc)
            continue

        # get pattern flag
        pattern_flag = pattern_flags[i]
        print(pattern)

        # train and evaluate pattern
        results_embedding_, results_features_ = train_evaluate_ml_models(
            pattern_name=pattern_flag,
            df=df_real,
            embeddings=embeddings)
        results_embedding[pattern] = results_embedding_
        results_features[pattern] = results_features_

        # verbose f1-score
        score_rf_embedding = np.mean(
            results_embedding_["RandomForestClassifier"]["f1_scores"])
        total_rf_embedding.append(score_rf_embedding)
        score_svc_embedding = np.mean(
            results_embedding_["SVC"]["f1_scores"])
        total_svc_embedding.append(score_svc_embedding)
        score_dummy_embedding = np.mean(
            results_embedding_["DummyClassifier"]["f1_scores"])
        score_rf_features = np.mean(
            results_features_["RandomForestClassifier"]["f1_scores"])
        total_rf_features.append(score_rf_features)
        score_svc_features = np.mean(
            results_features_["SVC"]["f1_scores"])
        total_svc_features.append(score_svc_features)
        score_dummy_features = np.mean(
            results_features_["DummyClassifier"]["f1_scores"])
        print("\r RandomForestClassifier: {0:0.3f} | {1:0.3f}"
              .format(score_rf_embedding, score_rf_features))
        print("\r SVC: {0:0.3f} | {1:0.3f}"
              .format(score_svc_embedding, score_svc_features))
        print("\r DummyClassifier: {0:0.3f} | {1:0.3f}"
              .format(score_dummy_embedding, score_dummy_features))
        print()

        # (re)train best models for learned features
        pipeline_rf, pipeline_svc = extract_best_ml_models(
            pattern_name=pattern_flag,
            df=df_real,
            embeddings=embeddings,
            d_results=results_embedding_)

        # save models
        path = os.path.join(
            log_directory, "pipeline_rf_{0}.joblib".format(pattern))
        joblib.dump(pipeline_rf, path)
        path = os.path.join(
            log_directory, "pipeline_svc_{0}.joblib".format(pattern))
        joblib.dump(pipeline_svc, path)

        # compute probabilities
        probability_rf = pipeline_rf.predict_proba(embeddings)[:, 1]
        probability_rf = probability_rf[:, np.newaxis]
        probabilities_rf.append(probability_rf)
        probability_svc = pipeline_svc.predict_proba(embeddings)[:, 1]
        probability_svc = probability_svc[:, np.newaxis]
        probabilities_svc.append(probability_svc)

    # save results
    add_strides = False
    if "Strides" in base_model_name:
        add_strides = True
    multiscale = False
    if "MS" in base_model_name:
        multiscale = True
    trainable_variables = get_trainable_variables(model)
    df_results = pd.DataFrame()
    df_results.loc[:, "model_embedding"] = [base_model_name] * 2
    df_results.loc[:, "cluster"] = [add_cluster] * 2
    df_results.loc[:, "morphology"] = [add_morphology] * 2
    df_results.loc[:, "distance"] = [add_distance] * 2
    df_results.loc[:, "nb_inputs"] = [nb_inputs] * 2
    df_results.loc[:, "inputs_alignment"] = [inputs_alignment] * 2
    df_results.loc[:, "features_alignment"] = [features_alignment] * 2
    df_results.loc[:, "strides"] = [add_strides] * 2
    df_results.loc[:, "multiscale"] = [multiscale] * 2
    df_results.loc[:, "model_ml"] = ["rf", 'svc']
    df_results.loc[:, "f1_score_embedding"] = [
        np.mean(total_rf_embedding),
        np.mean(total_svc_embedding)]
    df_results.loc[:, "f1_score_features"] = [
        np.mean(total_rf_features),
        np.mean(total_svc_features)]
    df_results.loc[:, "ari_embedding"] = [rand_score_learned] * 2
    df_results.loc[:, "ari_features"] = [rand_score_manual] * 2
    df_results.loc[:, "trainable_variables"] = [trainable_variables] * 2
    df_results.loc[:, "nb_head"] = [nb_head] * 2
    df_results.loc[:, "k"] = [k] * 2
    df_results.loc[:, "latent_dimension"] = [latent_dimension] * 2
    df_results.loc[:, "directory"] = [log_directory.split("/")[-1]] * 2

    # save metadata dataframe
    path = os.path.join(log_directory, "results.csv")
    stack.save_data_to_csv(df_results, path)

    # format probabilities
    probabilities_rf = np.hstack(probabilities_rf)
    probabilities_svc = np.hstack(probabilities_svc)

    # add a default random predictions to plot
    probabilities_rf, _, _ = predict_default_random(
        probabilities=probabilities_rf,
        predictions=None,
        labels=None)
    probabilities_svc, _, _ = predict_default_random(
        probabilities=probabilities_svc,
        predictions=None,
        labels=None)

    # save results probabilities
    path = os.path.join(log_directory, "clf_probabilities_rf_real.npy")
    stack.save_array(probabilities_rf, path)
    path = os.path.join(log_directory, "clf_probabilities_svc_real.npy")
    stack.save_array(probabilities_svc, path)

    # plot UMAP
    path = os.path.join(
        plot_directory_embedding, "umap_real_probabilities_rf.png")
    plot_embedding_probability_real(
        embedding_2d=embedding_2d_umap,
        probabilities=probabilities_rf,
        df=df_real,
        path_output=path,
        extension="png")
    path = os.path.join(
        plot_directory_embedding, "umap_real_probabilities_rf.pdf")
    plot_embedding_probability_real(
        embedding_2d=embedding_2d_umap,
        probabilities=probabilities_rf,
        df=df_real,
        path_output=path,
        extension="pdf")
    path = os.path.join(
        plot_directory_embedding, "umap_real_probabilities_svc.png")
    plot_embedding_probability_real(
        embedding_2d=embedding_2d_umap,
        probabilities=probabilities_svc,
        df=df_real,
        path_output=path,
        extension="png")
    path = os.path.join(
        plot_directory_embedding, "umap_real_probabilities_svc.pdf")
    plot_embedding_probability_real(
        embedding_2d=embedding_2d_umap,
        probabilities=probabilities_svc,
        df=df_real,
        path_output=path,
        extension="pdf")

    # plot boxplot scores
    plot_directory_boxplot = os.path.join(plot_directory, "boxplot")
    os.mkdir(plot_directory_boxplot)
    plot_boxplot_balanced_accuracy(
        results_embedding=results_embedding,
        results_features=results_features,
        plot_directory=plot_directory_boxplot)
    plot_boxplot_f1(
        results_embedding=results_embedding,
        results_features=results_features,
        plot_directory=plot_directory_boxplot)
    plot_boxplot_precision(
        results_embedding=results_embedding,
        results_features=results_features,
        plot_directory=plot_directory_boxplot)
    plot_boxplot_recall(
        results_embedding=results_embedding,
        results_features=results_features,
        plot_directory=plot_directory_boxplot)
    plot_boxplot_auc(
        results_embedding=results_embedding,
        results_features=results_features,
        plot_directory=plot_directory_boxplot)

    #  time control
    end_script(start_time)

    print("Script completed!")
