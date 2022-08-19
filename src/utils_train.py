# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Training (utility functions).
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

import bigfish.plot as plot

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import adjusted_rand_score

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


# ### Training (deep learning) ###

def define_callbacks(log_directory_experiment, model, patience=5):
    # define callbacks
    callback_early_stop = tf.keras.callbacks.EarlyStopping(
        patience=patience,
        restore_best_weights=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_directory_experiment)
    path_checkpoint = os.path.join(log_directory_experiment, "checkpoint")
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        path_checkpoint,
        save_weights_only=True,
        save_best_only=True)
    path_log = os.path.join(log_directory_experiment, "history.csv")
    callback_csv = tf.keras.callbacks.CSVLogger(
        path_log,
        separator=";")

    # enlist callbacks
    callbacks = [callback_early_stop, callback_tensorboard,
                 callback_checkpoint, callback_csv]
    callbacks_list = tf.keras.callbacks.CallbackList(
        callbacks=callbacks,
        model=model)

    return callbacks_list


class LRSchedule(LearningRateSchedule):

    def __init__(self,
                 initial_learning_rate=0.001,
                 decay_steps=100,
                 decay_rate=0.5,
                 min_learning_rate=0.00001,
                 **kwargs):
        super(LRSchedule, self).__init__()

        # initialize parameters
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        decay = self.decay_rate ** (step / self.decay_steps)
        lr = self.initial_learning_rate * decay
        lr = tf.math.maximum(lr, self.min_learning_rate)

        return lr

    def get_config(self):
        config = super(LRSchedule, self).get_config()
        config.update({
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'decay_rate': self.decay_rate,
            'min_learning_rate': self.min_learning_rate})

        return config


# ### Training (machine learning) ###

def initialize_ml_models():
    # initialize different pipelines and parameter grids
    names = []
    pipelines = []
    param_grids = []

    # define pipeline RandomForestClassifier
    scaler_rfc = StandardScaler()
    clf_rfc = RandomForestClassifier()
    pipeline_rfc = Pipeline([
        ("scaler_rfc", scaler_rfc),
        ("clf_rfc", clf_rfc)])
    param_grid_rfc = {
        "clf_rfc__n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "clf_rfc__criterion": ["gini", "entropy"],
        "clf_rfc__max_depth": [3, 6, 9]}
    names.append("RandomForestClassifier")
    pipelines.append(pipeline_rfc)
    param_grids.append(param_grid_rfc)

    # define pipeline SVC
    scaler_svc = StandardScaler()
    clf_svc = SVC()
    pipeline_svc = Pipeline([
        ("scaler_svc", scaler_svc),
        ("clf_svc", clf_svc)])
    param_grid_svc = {
        "clf_svc__C": np.logspace(-3, 2, 6),
        "clf_svc__gamma": np.logspace(-3, 2, 6),
        "clf_svc__kernel": ["linear", "rbf"]}
    names.append("SVC")
    pipelines.append(pipeline_svc)
    param_grids.append(param_grid_svc)

    # define pipeline DummyClassifier
    scaler_dc = StandardScaler()
    clf_dc = DummyClassifier()
    pipeline_dc = Pipeline([
        ("scaler_dc", scaler_dc),
        ("clf_dc", clf_dc)])
    param_grid_dc = {}
    names.append("DummyClassifier")
    pipelines.append(pipeline_dc)
    param_grids.append(param_grid_dc)

    return names, pipelines, param_grids


def gridsearch_model(
        X_train,
        y_train,
        X_test,
        y_test,
        n_inner,
        model_name,
        pipeline,
        param_grid,
        d_results):
    # gridsearch embedding (train-validation)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        n_jobs=-1,
        cv=n_inner)
    search.fit(X_train, y_train)
    best_parameters = search.best_params_
    d_results[model_name]["best_parameters"].append(best_parameters)
    best_pipeline = search.best_estimator_

    # evaluation embedding (test)
    y_pred = best_pipeline.predict(X_test)
    if model_name == "SVC":
        y_proba = best_pipeline.decision_function(X_test)
    else:
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    accuracy = balanced_accuracy_score(
        y_true=y_test,
        y_pred=y_pred)
    d_results[model_name]["balanced_accuracy_scores"].append(accuracy)
    f1 = f1_score(
        y_true=y_test,
        y_pred=y_pred)
    d_results[model_name]["f1_scores"].append(f1)
    precision = precision_score(
        y_true=y_test,
        y_pred=y_pred,
        zero_division=0)
    d_results[model_name]["precision_scores"].append(precision)
    recall = recall_score(
        y_true=y_test,
        y_pred=y_pred,
        zero_division=0)
    d_results[model_name]["recall_scores"].append(recall)
    auc_roc = roc_auc_score(
        y_true=y_test,
        y_score=y_proba)
    d_results[model_name]["auc_roc_scores"].append(auc_roc)

    return d_results


def train_evaluate_ml_models(pattern_name, df, embeddings, verbose=False):
    # filter dataframe
    mask_annotated = np.array(df.loc[:, "annotated"])
    df_annotated = df.loc[mask_annotated, :]

    # initialize results
    model_names, _, _ = initialize_ml_models()
    results_embedding = {}
    results_features = {}
    for model_name in model_names:
        results_embedding[model_name] = {
            "best_parameters": [],
            "balanced_accuracy_scores": [],
            "f1_scores": [],
            "precision_scores": [],
            "recall_scores": [],
            "auc_roc_scores": []}
        results_features[model_name] = {
            "best_parameters": [],
            "balanced_accuracy_scores": [],
            "f1_scores": [],
            "precision_scores": [],
            "recall_scores": [],
            "auc_roc_scores": []}

    # get features
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
    X_features = df_annotated.loc[:, features].to_numpy()
    X_embedding = embeddings[mask_annotated, ...].copy()

    # get labels
    y = np.array(df_annotated.loc[:, pattern_name]).astype(np.int64)

    # count positive and negative labels
    if verbose:
        n = y.size
        n_positive = np.sum(y == 1)
        p_positive = n_positive / n * 100
        n_negative = n - n_positive
        p_negative = 100 - p_positive
        print("\r {0} positives ({1:0.1f}%) | {2} negatives ({3:0.1f}%)"
              .format(n_positive, p_positive, n_negative, p_negative))

    # nested cv-fold (outer loop)
    sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2)
    for train_index, test_index in sss.split(X_embedding, y):

        # split train-test
        X_embedding_train = X_embedding[train_index]
        X_features_train = X_features[train_index]
        X_embedding_test = X_embedding[test_index]
        X_features_test = X_features[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # count positive and negative labels
        if verbose:
            n = y_test.size
            n_positive = np.sum(y_test == 1)
            p_positive = n_positive / n * 100
            n_negative = n - n_positive
            p_negative = 100 - p_positive
            print("\r     {0} ({1:0.1f}%) | {2} ({3:0.1f}%) | {4} | {5}..."
                  .format(n_positive, p_positive,
                          n_negative, p_negative, n, test_index[:10]))

        # ### Embeddings ###

        # initialize models
        names, pipelines, param_grids = initialize_ml_models()

        # loop over models
        for (model_name, pipeline, param_grid) in zip(
                names, pipelines, param_grids):

            # perform a gridsearch
            results_embedding = gridsearch_model(
                X_train=X_embedding_train,
                y_train=y_train,
                X_test=X_embedding_test,
                y_test=y_test,
                n_inner=4,
                model_name=model_name,
                pipeline=pipeline,
                param_grid=param_grid,
                d_results=results_embedding)

        # ### Features ###

        # initialize models
        names, pipelines, param_grids = initialize_ml_models()

        # loop over models
        for (model_name, pipeline, param_grid) in zip(
                names, pipelines, param_grids):

            # perform a gridsearch
            results_features = gridsearch_model(
                X_train=X_features_train,
                y_train=y_train,
                X_test=X_features_test,
                y_test=y_test,
                n_inner=4,
                model_name=model_name,
                pipeline=pipeline,
                param_grid=param_grid,
                d_results=results_features)

    return results_embedding, results_features


def extract_best_ml_models(pattern_name, df, embeddings, d_results):
    # filter dataframe
    mask_annotated = np.array(df.loc[:, "annotated"])
    df_annotated = df.loc[mask_annotated, :]

    # get features and labels
    X_embedding = embeddings[mask_annotated, ...].copy()
    y = np.array(df_annotated.loc[:, pattern_name]).astype(np.int64)

    # get parameters RandomForestClassifier
    best_parameters = d_results["RandomForestClassifier"]["best_parameters"]
    param_criterion = [
        param["clf_rfc__criterion"] for param in best_parameters]
    param_depth = [
        param["clf_rfc__max_depth"] for param in best_parameters]
    param_n_estimators = [
        param["clf_rfc__n_estimators"] for param in best_parameters]
    criterion = _most_frequent(param_criterion)
    mask_criterion = np.array(param_criterion) == criterion
    max_depth = _most_frequent(np.array(param_depth)[mask_criterion].tolist())
    mask_depth = (np.array(param_depth) == max_depth) & mask_criterion
    n_estimators = _most_frequent(
        np.array(param_n_estimators)[mask_depth].tolist())

    # define model RandomForestClassifier
    scaler_rf = StandardScaler()
    clf_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=13)
    pipeline_rf = Pipeline([
        ("scaler_rf", scaler_rf),
        ("clf_rf", clf_rf)])

    # get parameters SVC
    best_parameters = d_results["SVC"]["best_parameters"]
    param_kernel = [param["clf_svc__kernel"] for param in best_parameters]
    param_c = [param["clf_svc__C"] for param in best_parameters]
    param_gamma = [param["clf_svc__gamma"] for param in best_parameters]
    kernel = _most_frequent(param_kernel)
    mask_kernel = np.array(param_kernel) == kernel
    c = _most_frequent(np.array(param_c)[mask_kernel].tolist())
    mask_c = (np.array(param_c) == c) & mask_kernel
    gamma = _most_frequent(np.array(param_gamma)[mask_c].tolist())

    # define model SVC
    scaler_svc = StandardScaler()
    clf_svc = SVC(
        C=c,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=13)
    pipeline_svc = Pipeline([
        ("scaler_svc", scaler_svc),
        ("clf_svc", clf_svc)])

    # train models
    pipeline_rf.fit(X_embedding, y)
    pipeline_svc.fit(X_embedding, y)

    return pipeline_rf, pipeline_svc


def _most_frequent(l):
    return max(set(l), key=l.count)


# ### Training (classification) ###

@tf.function(experimental_relax_shapes=True)
def train_step_clf(inputs, labels, model, loss_fn, optimizer):
    # forward pass
    with tf.GradientTape() as tape:
        output = model(inputs, training=True)
        loss_value = loss_fn(labels, output)
        internal_loss = tf.reduce_sum(model.losses)
        loss_value = tf.add(loss_value, internal_loss)

    # get gradients and update weights
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return output, loss_value, internal_loss


@tf.function(experimental_relax_shapes=True)
def val_step_clf(inputs, labels, model, loss_fn):
    # forward pass
    output = model(inputs, training=False)
    loss_value = loss_fn(labels, output)
    internal_loss = tf.reduce_sum(model.losses)
    loss_value = tf.add(loss_value, internal_loss)

    return output, loss_value, internal_loss


# ### Training (pretext task) ###

@tf.function(experimental_relax_shapes=True)
def train_step_pretext(
        inputs,
        labels_pattern,
        labels_cluster,
        labels_morphology,
        labels_distance,
        model,
        losses_fn,
        optimizer):
    # forward pass
    with tf.GradientTape() as tape:
        output = model(inputs, training=True)
        internal_loss = tf.reduce_sum(model.losses)
        loss_pattern = losses_fn[0](labels_pattern, output[0])
        loss_cluster = losses_fn[1](labels_cluster, output[1])
        loss_morphology = losses_fn[2](labels_morphology, output[2])
        loss_distance = losses_fn[3](labels_distance, output[3])
        loss_value = (internal_loss
                      + 0.5 * loss_pattern
                      + 0.1665 * loss_cluster
                      + 0.1665 * loss_morphology
                      + 0.1665 * loss_distance)

    # get gradients and update weights
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return output, loss_value, internal_loss


@tf.function(experimental_relax_shapes=True)
def val_step_pretext(
        inputs,
        labels_pattern,
        labels_cluster,
        labels_morphology,
        labels_distance,
        model,
        losses_fn):
    # forward pass
    output = model(inputs, training=False)
    internal_loss = tf.reduce_sum(model.losses)
    loss_pattern = losses_fn[0](labels_pattern, output[0])
    loss_cluster = losses_fn[1](labels_cluster, output[1])
    loss_morphology = losses_fn[2](labels_morphology, output[2])
    loss_distance = losses_fn[3](labels_distance, output[3])
    loss_value = (internal_loss
                  + 0.5 * loss_pattern
                  + 0.1665 * loss_cluster
                  + 0.1665 * loss_morphology
                  + 0.1665 * loss_distance)

    return output, loss_value, internal_loss


# ### Evaluation (general) ###

def predict_default_random(probabilities, predictions, labels):
    # add random probabilities
    if probabilities is not None:
        random_probabilities = 1 - probabilities.max(axis=1)
        random_probabilities = random_probabilities[:, np.newaxis]
        probabilities = np.hstack([random_probabilities, probabilities])

    # add random predictions
    if predictions is not None:
        mask_random = np.all(predictions == 0, axis=1)
        random_predictions = np.zeros(len(predictions), dtype=np.int64)
        random_predictions[mask_random] = 1
        random_predictions = random_predictions[:, np.newaxis]
        predictions = np.hstack([random_predictions, predictions])

    # add random labels
    if labels is not None:
        mask_random = np.all(labels == 0, axis=1)
        random_labels = np.zeros(len(labels), dtype=np.int64)
        random_labels[mask_random] = 1
        random_labels = random_labels[:, np.newaxis]
        labels = np.hstack([random_labels, labels])

    return probabilities, predictions, labels


def evaluation_embedding(trained_model_embedding, dataset):
    # loop over batches
    names = []
    embeddings = []
    probabilities = []
    predictions = []
    for i, x in enumerate(dataset):

        # get instance name
        name = x["name_sample"].numpy()
        name = name[0][0].decode("utf-8")
        names.append(name)

        # get results
        embedding, probability = trained_model_embedding.predict_on_batch(x)
        embeddings.append(embedding)
        probabilities.append(probability)
        prediction = (probability > 0.5).astype(np.int64)
        predictions.append(prediction)

    # format results
    embeddings = np.concatenate(embeddings)
    probabilities = np.concatenate(probabilities)
    predictions = np.concatenate(predictions)

    return names, embeddings, probabilities, predictions


# ### Evaluation (classification) ###

def evaluation_clf(trained_model, dataset):
    # loop over batches
    probabilities = []
    labels = []
    for x, y in dataset:

        # get ground truth
        y_true = y["label"].numpy().astype(np.int64)
        labels.append(y_true)

        # get probabilities
        probability = trained_model.predict_on_batch(x)
        probabilities.append(probability)

    # format results
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    indices_max = np.argmax(probabilities, axis=1)
    predictions = np.zeros(probabilities.shape, dtype=np.int64)
    for i in range(len(predictions)):
        predictions[i, indices_max[i]] = 1

    # compute metrics
    metrics = compute_global_metrics_clf(
        y_true=labels,
        y_pred=predictions)

    return metrics, probabilities, predictions, labels


def compute_global_metrics_clf(y_true, y_pred):
    # compute metrics
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        zero_division=0)
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        zero_division=0)
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        zero_division=0)

    # format results
    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall}

    return metrics


# ### Evaluation (pretext task) ###

def evaluation_pretext(trained_model, dataset, return_pattern_results=False):
    # loop over batches
    probabilities_pattern = []
    predictions_pattern = []
    labels_pattern = []
    predictions_cluster = []
    labels_cluster = []
    predictions_morphology = []
    labels_morphology = []
    predictions_distance = []
    labels_distance = []
    for x, y in dataset:

        # get ground truth
        y_true_pattern = y["label_pattern"].numpy().astype(np.int64)
        labels_pattern.append(y_true_pattern)
        y_true_cluster = y["label_cluster"].numpy().astype(np.int64)
        labels_cluster.append(y_true_cluster[0])
        y_true_morphology = y["label_morphology"].numpy().astype(np.int64)
        labels_morphology.append(y_true_morphology[0])
        y_true_distance = y["label_distance"].numpy().astype(np.float32)
        labels_distance.append(y_true_distance[0])

        # get predictions
        output = trained_model.predict_on_batch(x)
        probability_pattern = output[0]  # (1, 8)
        probabilities_pattern.append(probability_pattern)
        indices_max = np.argmax(probability_pattern, axis=1)
        prediction_pattern = np.zeros(
            probability_pattern.shape, dtype=np.int64)
        for i in range(len(prediction_pattern)):
            prediction_pattern[i, indices_max[i]] = 1
        predictions_pattern.append(prediction_pattern)
        probability_cluster = output[1][0]  # (nb_points, 1)
        prediction_cluster = (probability_cluster > 0.5).astype(np.int64)
        predictions_cluster.append(prediction_cluster)
        probability_morphology = output[2][0]  # (nb_points, 3)
        indices_max = np.argmax(probability_morphology, axis=1)
        prediction_morphology = np.zeros(
            probability_morphology.shape, dtype=np.int64)
        for i in range(len(prediction_morphology)):
            prediction_morphology[i, indices_max[i]] = 1
        predictions_morphology.append(prediction_morphology)
        prediction_distance = output[3][0]  # (nb_points, 2)
        predictions_distance.append(prediction_distance)

    # format results
    probabilities_pattern = np.concatenate(probabilities_pattern)
    predictions_pattern = np.concatenate(predictions_pattern)
    labels_pattern = np.concatenate(labels_pattern)
    predictions_cluster = np.concatenate(predictions_cluster)
    labels_cluster = np.concatenate(labels_cluster)
    predictions_morphology = np.concatenate(predictions_morphology)
    labels_morphology = np.concatenate(labels_morphology)
    predictions_distance = np.concatenate(predictions_distance)
    labels_distance = np.concatenate(labels_distance)

    # compute metrics and loss
    metrics = compute_global_metrics_pretext(
        y_true_pattern=labels_pattern,
        y_pred_pattern=predictions_pattern,
        y_true_cluster=labels_cluster,
        y_pred_cluster=predictions_cluster,
        y_true_morphology=labels_morphology,
        y_pred_morphology=predictions_morphology,
        y_true_distance=labels_distance,
        y_pred_distance=predictions_distance)

    if return_pattern_results:
        return (metrics,
                probabilities_pattern, predictions_pattern, labels_pattern)
    else:
        return metrics


def compute_global_metrics_pretext(
        y_true_pattern,
        y_pred_pattern,
        y_true_cluster,
        y_pred_cluster,
        y_true_morphology,
        y_pred_morphology,
        y_true_distance,
        y_pred_distance):
    # compute metrics pattern
    f1_pattern = f1_score(
        y_true=y_true_pattern,
        y_pred=y_pred_pattern,
        average="macro",
        zero_division=0)
    precision_pattern = precision_score(
        y_true=y_true_pattern,
        y_pred=y_pred_pattern,
        average="macro",
        zero_division=0)
    recall_pattern = recall_score(
        y_true=y_true_pattern,
        y_pred=y_pred_pattern,
        average="macro",
        zero_division=0)

    # compute metrics cluster
    f1_cluster = f1_score(
        y_true=y_true_cluster,
        y_pred=y_pred_cluster,
        zero_division=0)
    precision_cluster = precision_score(
        y_true=y_true_cluster,
        y_pred=y_pred_cluster,
        zero_division=0)
    recall_cluster = recall_score(
        y_true=y_true_cluster,
        y_pred=y_pred_cluster,
        zero_division=0)

    # compute metrics morphology
    f1_morphology = f1_score(
        y_true=y_true_morphology,
        y_pred=y_pred_morphology,
        average="macro",
        zero_division=0)
    precision_morphology = precision_score(
        y_true=y_true_morphology,
        y_pred=y_pred_morphology,
        average="macro",
        zero_division=0)
    recall_morphology = recall_score(
        y_true=y_true_morphology,
        y_pred=y_pred_morphology,
        average="macro",
        zero_division=0)

    # compute metrics distance
    mae_distance_cell = np.mean(
        np.abs(y_true_distance[:, 0] - y_pred_distance[:, 0]), axis=-1)
    mae_distance_nuc = np.mean(
        np.abs(y_true_distance[:, 1] - y_pred_distance[:, 1]), axis=-1)

    # format results
    metrics = {
        "f1_pattern": f1_pattern,
        "precision_pattern": precision_pattern,
        "recall_pattern": recall_pattern,
        "f1_cluster": f1_cluster,
        "precision_cluster": precision_cluster,
        "recall_cluster": recall_cluster,
        "f1_morphology": f1_morphology,
        "precision_morphology": precision_morphology,
        "recall_morphology": recall_morphology,
        "mae_distance_cell": mae_distance_cell,
        "mae_distance_nuc": mae_distance_nuc}

    return metrics


# ### Plots ###

def plot_confusion_matrix(
        y_true,
        y_pred,
        path_output=None,
        extension=None,
        show=False):
    # format arrays
    y_true = np.nonzero(y_true)[1]
    y_pred = np.nonzero(y_pred)[1]

    # pattern names
    patterns_name = [
        "random", "foci", "intranuclear", "extranuclear",
        "nuclear edge", "perinuclear", "cell edge", "pericellular"]

    # compute confusion matrix
    m = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=[i for i in range(len(patterns_name))],
        normalize="true")

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    display = ConfusionMatrixDisplay(m, display_labels=patterns_name)
    display.plot(xticks_rotation="vertical", ax=ax, colorbar=False)
    ax.set_xlabel("Predicted label", fontweight="bold", fontsize=15)
    ax.set_ylabel("True label", fontweight="bold", fontsize=15)

    # colobar
    ax = plt.gca()
    im = ax.get_children()[-2]
    plt.colorbar(im, ax=ax, fraction=0.045, shrink=0.9)

    # format plot
    plt.tight_layout()

    # output
    if path_output is not None:
        plot.save_plot(path_output, extension)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_confusion_matrices_binary(
        y_true,
        y_pred,
        y_proba,
        path_output=None,
        extension=None,
        show=False):
    # compute confusion matrices
    matrices_confusion = multilabel_confusion_matrix(
        y_true=y_true, y_pred=y_pred)

    # define patterns
    patterns_name = ["random", "foci", "intranuclear", "nuclear edge",
                     "perinuclear", "cell edge", "protrusion"]

    # plot confusion matrices
    im = None
    fig, ax = plt.subplots(2, 4, figsize=(18.5, 8))
    for i, m in enumerate(matrices_confusion):

        # compute metrics
        pattern = patterns_name[i]
        row, col = i % 2, i // 2
        tn, fp, fn, tp = m.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / max((tp + fp), 1)
        recall = tp / max((tp + fn), 1)

        # normalize
        sum_of_rows = m.sum(axis=1)
        normalized_m = m / sum_of_rows[:, np.newaxis]

        # plot
        im = ax[row, col].imshow(normalized_m)

        # subplots title and axes
        ax[row, col].set_title(
            "{0} ({1:0.3f} | {2:0.3f} | {3:0.3f})".format(
                pattern, acc, precision, recall),
            fontweight="bold", fontsize=12)
        if row == 1 or col == 3:
            ax[row, col].set_xlabel(
                "Predicted label", fontweight="bold", fontsize=10)
        if col == 0:
            ax[row, col].set_ylabel(
                "True label", fontweight="bold", fontsize=10)

        # subplots text
        cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)
        for (m_row, m_col), value in np.ndenumerate(normalized_m):
            value = round(value, 3)
            color = cmap_max if value < 0.5 else cmap_min
            ax[row, col].text(m_col, m_row, value, ha='center', va='center',
                              fontweight="bold", fontsize=15, color=color)

    # precision-recall curve for the last subplot
    for i, pattern in enumerate(patterns_name):
        precision_pattern, recall_pattern, _ = precision_recall_curve(
            y_true=y_true[:, i],
            probas_pred=y_proba[:, i])
        ax[1, 3].plot(recall_pattern, precision_pattern, label=pattern)
    ax[1, 3].legend(loc="lower left")
    ax[1, 3].set_xlabel("Recall", fontweight="bold", fontsize=10)
    ax[1, 3].set_ylabel("Precision", fontweight="bold", fontsize=10)
    ax[1, 3].set_title("Precision-Recall curve", fontweight="bold",
                       fontsize=12)

    # format subplots ticks
    plt.setp(ax,
             xticks=[0, 1], xticklabels=[0, 1],
             yticks=[0, 1], yticklabels=[0, 1])

    # frame
    plt.tight_layout()

    # colobar
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.9)
    cbar.set_ticks(np.arange(0, 1.1, 0.2))
    cbar.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=15)

    # output
    if path_output is not None:
        plot.save_plot(path_output, extension)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_embedding_real(
        embedding_2d,
        df,
        figsize=(12, 12),
        legend="inside",
        path_output=None,
        extension=None,
        show=False):
    # parameters
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    patterns_name = ['intranuclear', 'nuclear', 'perinuclear', "protrusion",
                     'foci', "random"]
    default_color = "#d9d9d9"

    # get labels for the embedding
    labels = list(df["label"])
    unique_labels = list(set(labels))
    encoder_label = LabelEncoder()
    encoder_label.fit(unique_labels)
    labels_num = encoder_label.transform(labels)

    # plot embedding
    plt.figure(figsize=figsize)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], label="unlabelled",
                c=default_color, alpha=0.7,
                marker="o", s=50)

    # color annotated observations
    for i_pattern, pattern_name in enumerate(patterns_name):
        colors_pattern = colors[i_pattern]
        label_num = encoder_label.transform([pattern_name])[0]
        plt.scatter(embedding_2d[labels_num == label_num, 0],
                    embedding_2d[labels_num == label_num, 1],
                    label=pattern_name, c=colors_pattern, alpha=1,
                    marker="o", s=150)

    # legend and ticks
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    if legend == "inside":
        plt.legend(prop={'size': 15}, loc='lower right')
    elif legend is None:
        pass
    else:
        plt.legend(prop={'size': 15}, loc='center left',
                   bbox_to_anchor=(1, 0.5))
    plt.axis("off")
    plt.tight_layout()

    # output
    if path_output is not None:
        plot.save_plot(path_output, extension)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_embedding_probability_real(
        embedding_2d,
        probabilities,
        df,
        path_output=None,
        extension=None,
        show=False):
    # ### probabilities ###

    # parameters
    patterns_name = ["random", "foci", "intranuclear", "nuclear edge",
                     "perinuclear", "cell edge", "protrusion"]
    colors = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys", "RdPu"]

    # subplots
    fig, ax = plt.subplots(4, 2, figsize=(12, 21))

    # remove annotated cell from the plot
    mask_no_annotation = ~np.array(df.loc[:, "annotated"])

    for i, pattern in enumerate(patterns_name):

        # get coordinates and colors
        row, col = i // 2, i % 2
        color = colors[i]
        x = embedding_2d[mask_no_annotation].copy()
        probability = probabilities[mask_no_annotation, i].copy()
        permutation = probability.argsort()
        probability_sorted = probability[permutation]
        x_sorted = x[permutation]

        # scatter plot
        ax[row, col].scatter(
            x_sorted[:, 0], x_sorted[:, 1],
            c=probability_sorted, cmap=color,
            marker="o", s=50)

        # legend and ticks
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_xlabel(pattern, fontweight="bold",
                                fontsize=10)

    # ### annotation ###

    # parameters
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    patterns_name = ['intranuclear', 'nuclear', 'perinuclear', "protrusion",
                     'foci', "random"]
    default_color = "#d9d9d9"

    # get labels for the embedding
    labels = list(df["label"])
    unique_labels = list(set(labels))
    encoder_label = LabelEncoder()
    encoder_label.fit(unique_labels)
    labels_num = encoder_label.transform(labels)

    # plot embedding
    ax[3, 1].scatter(
        embedding_2d[:, 0], embedding_2d[:, 1], label="unlabelled",
        c=default_color, alpha=0.7, marker="o", s=30)

    # color annotated observations
    for i, pattern in enumerate(patterns_name):
        color = colors[i]
        label_num = encoder_label.transform([pattern])[0]
        ax[3, 1].scatter(
            embedding_2d[labels_num == label_num, 0],
            embedding_2d[labels_num == label_num, 1],
            label=pattern, c=color, alpha=1,
            marker="o", s=50)

    # legend and ticks
    ax[3, 1].set_xticks([])
    ax[3, 1].set_yticks([])
    ax[3, 1].legend()
    ax[3, 1].set_xlabel("Manual annotation", fontweight="bold", fontsize=10)

    # format plot
    plt.tight_layout()

    # output
    if path_output is not None:
        plot.save_plot(path_output, extension)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_embedding_genes_real(
        embedding_2d,
        df,
        figsize=(12, 12),
        legend="inside",
        path_output=None,
        extension=None,
        show=False):
    # parameters
    genes_random = [
        "KIF20B", "MYO18A", "MYSNE2", "PLEC"]
    color_random = ["#a65628"] * len(genes_random)
    marker_random = ["o", "v", "s", "p"]
    genes_foci = [
        "HMMR", "CEP170P1", "CTNNB1", "AURKA", "CRKL",
        "DYNC1H1", "BUB1", "PAK2"]
    color_foci = ["#ff7f00"] * len(genes_foci)
    marker_foci = ["o", "v", "s", "p", "D", "P", "*", "X"]
    genes_intranuclear = [
        "MYH3", "CEP192"]
    color_intranuclear = ["#e41a1c"] * len(genes_intranuclear)
    marker_intranuclear = ["o", "v"]
    genes_nuclear_edge = [
        "ASPM", "SPEN"]
    color_nuclear_edge = ["#377eb8"] * len(genes_nuclear_edge)
    marker_nuclear_edge = ["o", "v"]
    genes_perinuclear = [
        "AP1S2", "ATP6A2", "AKAP9", "AKAP1", "HSP90B1"]
    color_perinuclear = ["#4daf4a"] * len(genes_perinuclear)
    marker_perinuclear = ["o", "v", "s", "p", "D"]
    genes_cell_edge = [
        "FLNA"]
    color_cell_edge = ["#000000"] * len(genes_cell_edge)
    marker_cell_edge = ["o"]
    genes_protrusion = [
        "KIF1C", "KIF4A", "RAB13", "DYNLL2", "KIF5B"]
    color_protrusion = ["#984ea3"] * len(genes_protrusion)
    marker_protrusion = ["o", "v", "s", "p", "D"]
    unique_genes = (
                genes_random + genes_foci + genes_intranuclear
                + genes_nuclear_edge + genes_perinuclear + genes_cell_edge
                + genes_protrusion)
    unique_colors = (
                color_random + color_foci + color_intranuclear
                + color_nuclear_edge + color_perinuclear + color_cell_edge
                + color_protrusion)
    unique_markers = (
                marker_random + marker_foci + marker_intranuclear
                + marker_nuclear_edge + marker_perinuclear + marker_cell_edge
                + marker_protrusion)
    default_color = "#d9d9d9"

    # get genes for the embedding
    genes = list(df.loc[:, "gene"])
    encoder_gene = LabelEncoder()
    encoder_gene.fit(unique_genes)
    genes_num = encoder_gene.transform(genes)

    # remove cells treated with puromycin
    mask_puromycin = df.loc[:, "puromycin"] == 1
    genes_num[mask_puromycin] = -1

    # plot embedding
    plt.figure(figsize=figsize)
    plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        label="puromycin",
        c=default_color,
        alpha=0.7,
        marker="o",
        s=50)

    # plot embedding according to genes
    for i_gene, gene in enumerate(unique_genes):
        color_gene = unique_colors[i_gene]
        marker_gene = unique_markers[i_gene]
        gene_num = encoder_gene.transform([gene])[0]
        plt.scatter(
            embedding_2d[genes_num == gene_num, 0],
            embedding_2d[genes_num == gene_num, 1],
            label=gene,
            c=color_gene,
            marker=marker_gene,
            alpha=1,
            s=150)

    # legend and ticks
    plt.tick_params(
        labelcolor='none',
        top=False,
        bottom=False,
        left=False,
        right=False)
    if legend == "inside":
        plt.legend(prop={'size': 15}, loc='lower right')
    elif legend is None:
        pass
    else:
        plt.legend(
            prop={'size': 15},
            loc='center left',
            bbox_to_anchor=(1, 0.5))
    plt.axis("off")
    plt.tight_layout()

    # output
    if path_output is not None:
        plot.save_plot(path_output, extension)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_embedding_genes_by_pattern_real(
        embedding_2d,
        df,
        pattern,
        figsize=(12, 12),
        legend="inside",
        path_output=None,
        extension=None,
        show=False):
    # parameters
    genes_random = [
        "KIF20B", "MYO18A", "MYSNE2", "PLEC"]
    genes_foci = [
        "HMMR", "CEP170P1", "CTNNB1", "AURKA", "CRKL",
        "DYNC1H1", "BUB1", "PAK2"]
    genes_intranuclear = [
        "MYH3", "CEP192"]
    genes_nuclear_edge = [
        "ASPM", "SPEN"]
    genes_perinuclear = [
        "AP1S2", "ATP6A2", "AKAP9", "AKAP1", "HSP90B1"]
    genes_cell_edge = [
        "FLNA"]
    genes_protrusion = [
        "KIF1C", "KIF4A", "RAB13", "DYNLL2", "KIF5B"]
    unique_genes = (
                genes_random + genes_foci + genes_intranuclear
                + genes_nuclear_edge + genes_perinuclear + genes_cell_edge
                + genes_protrusion)
    d_genes = {
        "random": genes_random,
        "foci": genes_foci,
        "intranuclear": genes_intranuclear,
        "nuclear_edge": genes_nuclear_edge,
        "perinuclear": genes_perinuclear,
        "cell_edge": genes_cell_edge,
        "protrusion": genes_protrusion}
    colors = ["#e41a1c", "#377eb8", "#4daf4a",
              "#984ea3", "#ff7f00", "#a65628",
              "#ffff33", "#000000"]
    markers = ["^", "v", "s", "p", "D", "P", "*", "X"]
    default_color = "#d9d9d9"

    # get genes for the embedding
    genes = list(df.loc[:, "gene"])
    encoder_gene = LabelEncoder()
    encoder_gene.fit(unique_genes)
    genes_num = encoder_gene.transform(genes)

    # remove cells treated with puromycin
    mask_puromycin = df.loc[:, "puromycin"] == 1
    genes_num[mask_puromycin] = -1

    # plot embedding
    plt.figure(figsize=figsize)
    plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        label="other",
        c=default_color,
        alpha=0.7,
        marker="o",
        s=50)

    # plot embedding according to selected genes
    selected_genes = d_genes[pattern]
    for i_gene, gene in enumerate(selected_genes):
        color_gene = colors[i_gene]
        marker_gene = markers[i_gene]
        gene_num = encoder_gene.transform([gene])[0]
        plt.scatter(
            embedding_2d[genes_num == gene_num, 0],
            embedding_2d[genes_num == gene_num, 1],
            label=gene,
            c=color_gene,
            marker=marker_gene,
            alpha=0.7,
            s=100)

    # legend and ticks
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False)
    if legend == "inside":
        plt.legend(prop={'size': 15}, loc='lower right')
    elif legend is None:
        pass
    else:
        plt.legend(
            prop={'size': 15}, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis("off")
    plt.tight_layout()

    # output
    if path_output is not None:
        plot.save_plot(path_output, extension)
    if show:
        plt.show()
    else:
        plt.close()

    return


# TODO adapt for ML predictions
def plot_heatmap_real_gene(
        results_genes,
        count_genes,
        nb_classes,
        path_output=None,
        extension=None,
        show=False):
    # parameters
    genes = [
        "MYH3", "CEP192",
        "ATP6A2", "AP1S2", "AKAP9", "AKAP1", "HSP90B1",
        "SPEN", "ASPM",
        "DYNC1H1", "BUB1", "CTNNB1", "HMMR", "CEP170P1", "CRKL", "PAK2",
        "AURKA",
        "KIF1C", "KIF4A", "RAB13", "DYNLL2", "KIF5B",
        "FLNA",
        "KIF20B", "MYO18A", "MYSNE2", "PLEC"]
    patterns_predicted = [
        "foci", "intranuclear", "nuclear edge", "perinuclear", "cell edge",
        "protrusion"]

    # build matrix
    m = np.zeros((len(results_genes), nb_classes))
    for i, gene in enumerate(genes):
        m[i, :] = results_genes[gene] / max(count_genes[gene], 1)

    # plot
    plt.figure(figsize=(10, 25))
    im = plt.imshow(m)

    # add text
    cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)
    for (row, col), value in np.ndenumerate(m):
        value = round(value, 3)
        color = cmap_max if value < 0.5 else cmap_min
        plt.text(col, row, value, ha='center', va='center', fontweight="bold",
                 fontsize=15, color=color)

    # format plot
    plt.title("Proportion of cells with a specific prediction",
              fontweight="bold", fontsize=15)
    plt.xticks([i for i in range(nb_classes)], patterns_predicted,
               fontweight="bold", fontsize=13, rotation=75)
    plt.yticks([i for i in range(len(genes))], genes, fontweight="bold",
               fontsize=13)
    plt.xlabel("Predictions", fontweight="bold", fontsize=15)
    plt.ylabel("Genes (Racha paper)", fontweight="bold", fontsize=15)

    # frame
    plt.tight_layout()

    # output
    if path_output is not None:
        plot.save_plot(path_output, extension)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_boxplot_balanced_accuracy(
        results_embedding,
        results_features,
        plot_directory,
        show=False):
    # parameters
    patterns = [
        "foci",
        "intranuclear", "nuclear_edge", "perinuclear",
        "protrusion"]

    # parameters plot
    colors = ["#bdbdbd", "#4daf4a", "#3182bd", "#5e3c99", "#fc9272"] * 2
    pattern_names = [
        "Foci", "Intranuclear", "Nuclear edge", "Perinuclear", "Protrusion"]
    model_names = ['RandomForestClassifier', 'SVC']

    # loop over models
    for model_name in model_names:

        # format results
        y_values = []
        for pattern in patterns:
            y_values.append(results_features[pattern][model_name][
                                "balanced_accuracy_scores"])
        for pattern in patterns:
            y_values.append(results_embedding[pattern][model_name][
                                "balanced_accuracy_scores"])

        # initialize plot
        plt.figure(figsize=(10, 4))

        # axes
        plt.axhline(y=0, c="black", lw=1, ls="dashed")
        plt.axhline(y=1, c="black", lw=1, ls="dashed")
        plt.axvline(x=5.5, c="gray", lw=1, ls="-")
        plt.ylim((-0.01, 1.01))
        plt.grid(which='major', axis='y', color="gray", alpha=0.6)

        # parameters plot
        boxprops = dict(
            linestyle='-',
            linewidth=2)
        flierprops = dict(
            marker='*',
            markerfacecolor='darkgray',
            markersize=10,
            markeredgecolor='darkgray')
        medianprops = dict(
            linestyle='-',
            linewidth=2,
            color='black')
        meanprops = dict(
            marker='D',
            markeredgecolor='black',
            markerfacecolor='firebrick')
        capprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')
        whiskerprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')

        # boxplot
        positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        boxes_pattern = plt.boxplot(
            x=y_values,
            positions=positions,
            widths=0.9,
            patch_artist=True,
            meanline=False,
            showmeans=True,
            showfliers=True,
            showbox=True,
            showcaps=True,
            whis=1.5,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            meanprops=meanprops,
            capprops=capprops,
            whiskerprops=whiskerprops)
        for i, patch in enumerate(boxes_pattern['boxes']):
            patch.set_facecolor(colors[i])

        # title, ticks and labels
        x_labels = ["Hand-crafted features", "Learned features"]
        x_ticks = [3, 8]
        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0,
                   fontweight="bold", fontsize=14)
        plt.yticks(fontweight="bold", fontsize=14)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(model_name, fontweight="bold", fontsize=20)

        # legend
        legend_elements = []
        for i in range(len(pattern_names)):
            element = mpatches.Patch(color=colors[i], label=pattern_names[i])
            legend_elements.append(element)
        plt.legend(handles=legend_elements, prop={'size': 15},
                   loc='lower left', framealpha=1)

        # format plot
        plt.tight_layout()

        # output
        path = os.path.join(
            plot_directory, "balanced_accuracy_{0}.png".format(model_name))
        plot.save_plot(path, "png")
        path = os.path.join(
            plot_directory, "balanced_accuracy_{0}.pdf".format(model_name))
        plot.save_plot(path, "pdf")

        # show plot
        if show:
            plt.show()
        else:
            plt.close()

    return


def plot_boxplot_f1(
        results_embedding,
        results_features,
        plot_directory,
        show=False):
    # parameters
    patterns = [
        "foci",
        "intranuclear", "nuclear_edge", "perinuclear",
        "protrusion"]

    # parameters plot
    colors = ["#bdbdbd", "#4daf4a", "#3182bd", "#5e3c99", "#fc9272"] * 2
    pattern_names = [
        "Foci", "Intranuclear", "Nuclear edge", "Perinuclear", "Protrusion"]
    model_names = ['RandomForestClassifier', 'SVC']

    # loop over models
    for model_name in model_names:

        # format results
        y_values = []
        for pattern in patterns:
            y_values.append(results_features[pattern][model_name][
                                "f1_scores"])
        for pattern in patterns:
            y_values.append(results_embedding[pattern][model_name][
                                "f1_scores"])

        # initialize plot
        plt.figure(figsize=(10, 4))

        # axes
        plt.axhline(y=0, c="black", lw=1, ls="dashed")
        plt.axhline(y=1, c="black", lw=1, ls="dashed")
        plt.axvline(x=5.5, c="gray", lw=1, ls="-")
        plt.ylim((-0.01, 1.01))
        plt.grid(which='major', axis='y', color="gray", alpha=0.6)

        # parameters plot
        boxprops = dict(
            linestyle='-',
            linewidth=2)
        flierprops = dict(
            marker='*',
            markerfacecolor='darkgray',
            markersize=10,
            markeredgecolor='darkgray')
        medianprops = dict(
            linestyle='-',
            linewidth=2,
            color='black')
        meanprops = dict(
            marker='D',
            markeredgecolor='black',
            markerfacecolor='firebrick')
        capprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')
        whiskerprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')

        # boxplot
        positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        boxes_pattern = plt.boxplot(
            x=y_values,
            positions=positions,
            widths=0.9,
            patch_artist=True,
            meanline=False,
            showmeans=True,
            showfliers=True,
            showbox=True,
            showcaps=True,
            whis=1.5,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            meanprops=meanprops,
            capprops=capprops,
            whiskerprops=whiskerprops)
        for i, patch in enumerate(boxes_pattern['boxes']):
            patch.set_facecolor(colors[i])

        # title, ticks and labels
        x_labels = ["Hand-crafted features", "Learned features"]
        x_ticks = [3, 8]
        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0,
                   fontweight="bold", fontsize=14)
        plt.yticks(fontweight="bold", fontsize=14)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(model_name, fontweight="bold", fontsize=20)

        # legend
        legend_elements = []
        for i in range(len(pattern_names)):
            element = mpatches.Patch(color=colors[i], label=pattern_names[i])
            legend_elements.append(element)
        plt.legend(handles=legend_elements, prop={'size': 15},
                   loc='lower left', framealpha=1)

        # format plot
        plt.tight_layout()

        # output
        path = os.path.join(
            plot_directory, "f1_{0}.png".format(model_name))
        plot.save_plot(path, "png")
        path = os.path.join(
            plot_directory, "f1_{0}.pdf".format(model_name))
        plot.save_plot(path, "pdf")

        # show plot
        if show:
            plt.show()
        else:
            plt.close()

    return


def plot_boxplot_precision(
        results_embedding,
        results_features,
        plot_directory,
        show=False):
    # parameters
    patterns = [
        "foci",
        "intranuclear", "nuclear_edge", "perinuclear",
        "protrusion"]

    # parameters plot
    colors = ["#bdbdbd", "#4daf4a", "#3182bd", "#5e3c99", "#fc9272"] * 2
    pattern_names = [
        "Foci", "Intranuclear", "Nuclear edge", "Perinuclear", "Protrusion"]
    model_names = ['RandomForestClassifier', 'SVC']

    # loop over models
    for model_name in model_names:

        # format results
        y_values = []
        for pattern in patterns:
            y_values.append(results_features[pattern][model_name][
                                "precision_scores"])
        for pattern in patterns:
            y_values.append(results_embedding[pattern][model_name][
                                "precision_scores"])

        # initialize plot
        plt.figure(figsize=(10, 4))

        # axes
        plt.axhline(y=0, c="black", lw=1, ls="dashed")
        plt.axhline(y=1, c="black", lw=1, ls="dashed")
        plt.axvline(x=5.5, c="gray", lw=1, ls="-")
        plt.ylim((-0.01, 1.01))
        plt.grid(which='major', axis='y', color="gray", alpha=0.6)

        # parameters plot
        boxprops = dict(
            linestyle='-',
            linewidth=2)
        flierprops = dict(
            marker='*',
            markerfacecolor='darkgray',
            markersize=10,
            markeredgecolor='darkgray')
        medianprops = dict(
            linestyle='-',
            linewidth=2,
            color='black')
        meanprops = dict(
            marker='D',
            markeredgecolor='black',
            markerfacecolor='firebrick')
        capprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')
        whiskerprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')

        # boxplot
        positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        boxes_pattern = plt.boxplot(
            x=y_values,
            positions=positions,
            widths=0.9,
            patch_artist=True,
            meanline=False,
            showmeans=True,
            showfliers=True,
            showbox=True,
            showcaps=True,
            whis=1.5,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            meanprops=meanprops,
            capprops=capprops,
            whiskerprops=whiskerprops)
        for i, patch in enumerate(boxes_pattern['boxes']):
            patch.set_facecolor(colors[i])

        # title, ticks and labels
        x_labels = ["Hand-crafted features", "Learned features"]
        x_ticks = [3, 8]
        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0,
                   fontweight="bold", fontsize=14)
        plt.yticks(fontweight="bold", fontsize=14)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(model_name, fontweight="bold", fontsize=20)

        # legend
        legend_elements = []
        for i in range(len(pattern_names)):
            element = mpatches.Patch(color=colors[i], label=pattern_names[i])
            legend_elements.append(element)
        plt.legend(handles=legend_elements, prop={'size': 15},
                   loc='lower left', framealpha=1)

        # format plot
        plt.tight_layout()

        # output
        path = os.path.join(
            plot_directory, "precision_{0}.png".format(model_name))
        plot.save_plot(path, "png")
        path = os.path.join(
            plot_directory, "precision_{0}.pdf".format(model_name))
        plot.save_plot(path, "pdf")

        # show plot
        if show:
            plt.show()
        else:
            plt.close()

    return


def plot_boxplot_recall(
        results_embedding,
        results_features,
        plot_directory,
        show=False):
    # parameters
    patterns = [
        "foci",
        "intranuclear", "nuclear_edge", "perinuclear",
        "protrusion"]

    # parameters plot
    colors = ["#bdbdbd", "#4daf4a", "#3182bd", "#5e3c99", "#fc9272"] * 2
    pattern_names = [
        "Foci", "Intranuclear", "Nuclear edge", "Perinuclear", "Protrusion"]
    model_names = ['RandomForestClassifier', 'SVC']

    # loop over models
    for model_name in model_names:

        # format results
        y_values = []
        for pattern in patterns:
            y_values.append(results_features[pattern][model_name][
                                "recall_scores"])
        for pattern in patterns:
            y_values.append(results_embedding[pattern][model_name][
                                "recall_scores"])

        # initialize plot
        plt.figure(figsize=(10, 4))

        # axes
        plt.axhline(y=0, c="black", lw=1, ls="dashed")
        plt.axhline(y=1, c="black", lw=1, ls="dashed")
        plt.axvline(x=5.5, c="gray", lw=1, ls="-")
        plt.ylim((-0.01, 1.01))
        plt.grid(which='major', axis='y', color="gray", alpha=0.6)

        # parameters plot
        boxprops = dict(
            linestyle='-',
            linewidth=2)
        flierprops = dict(
            marker='*',
            markerfacecolor='darkgray',
            markersize=10,
            markeredgecolor='darkgray')
        medianprops = dict(
            linestyle='-',
            linewidth=2,
            color='black')
        meanprops = dict(
            marker='D',
            markeredgecolor='black',
            markerfacecolor='firebrick')
        capprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')
        whiskerprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')

        # boxplot
        positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        boxes_pattern = plt.boxplot(
            x=y_values,
            positions=positions,
            widths=0.9,
            patch_artist=True,
            meanline=False,
            showmeans=True,
            showfliers=True,
            showbox=True,
            showcaps=True,
            whis=1.5,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            meanprops=meanprops,
            capprops=capprops,
            whiskerprops=whiskerprops)
        for i, patch in enumerate(boxes_pattern['boxes']):
            patch.set_facecolor(colors[i])

        # title, ticks and labels
        x_labels = ["Hand-crafted features", "Learned features"]
        x_ticks = [3, 8]
        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0,
                   fontweight="bold", fontsize=14)
        plt.yticks(fontweight="bold", fontsize=14)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(model_name, fontweight="bold", fontsize=20)

        # legend
        legend_elements = []
        for i in range(len(pattern_names)):
            element = mpatches.Patch(color=colors[i], label=pattern_names[i])
            legend_elements.append(element)
        plt.legend(handles=legend_elements, prop={'size': 15},
                   loc='lower left', framealpha=1)

        # format plot
        plt.tight_layout()

        # output
        path = os.path.join(
            plot_directory, "recall_{0}.png".format(model_name))
        plot.save_plot(path, "png")
        path = os.path.join(
            plot_directory, "recall_{0}.pdf".format(model_name))
        plot.save_plot(path, "pdf")

        # show plot
        if show:
            plt.show()
        else:
            plt.close()

    return


def plot_boxplot_auc(
        results_embedding,
        results_features,
        plot_directory,
        show=False):
    # parameters
    patterns = [
        "foci",
        "intranuclear", "nuclear_edge", "perinuclear",
        "protrusion"]

    # parameters plot
    colors = ["#bdbdbd", "#4daf4a", "#3182bd", "#5e3c99", "#fc9272"] * 2
    pattern_names = [
        "Foci", "Intranuclear", "Nuclear edge", "Perinuclear", "Protrusion"]
    model_names = ['RandomForestClassifier', 'SVC']

    # loop over models
    for model_name in model_names:

        # format results
        y_values = []
        for pattern in patterns:
            y_values.append(results_features[pattern][model_name][
                                "auc_roc_scores"])
        for pattern in patterns:
            y_values.append(results_embedding[pattern][model_name][
                                "auc_roc_scores"])

        # initialize plot
        plt.figure(figsize=(10, 4))

        # axes
        plt.axhline(y=0, c="black", lw=1, ls="dashed")
        plt.axhline(y=1, c="black", lw=1, ls="dashed")
        plt.axvline(x=5.5, c="gray", lw=1, ls="-")
        plt.ylim((-0.01, 1.01))
        plt.grid(which='major', axis='y', color="gray", alpha=0.6)

        # parameters plot
        boxprops = dict(
            linestyle='-',
            linewidth=2)
        flierprops = dict(
            marker='*',
            markerfacecolor='darkgray',
            markersize=10,
            markeredgecolor='darkgray')
        medianprops = dict(
            linestyle='-',
            linewidth=2,
            color='black')
        meanprops = dict(
            marker='D',
            markeredgecolor='black',
            markerfacecolor='firebrick')
        capprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')
        whiskerprops = dict(
            linestyle='-',
            linewidth=1.5,
            color='black')

        # boxplot
        positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        boxes_pattern = plt.boxplot(
            x=y_values,
            positions=positions,
            widths=0.9,
            patch_artist=True,
            meanline=False,
            showmeans=True,
            showfliers=True,
            showbox=True,
            showcaps=True,
            whis=1.5,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            meanprops=meanprops,
            capprops=capprops,
            whiskerprops=whiskerprops)
        for i, patch in enumerate(boxes_pattern['boxes']):
            patch.set_facecolor(colors[i])

        # title, ticks and labels
        x_labels = ["Hand-crafted features", "Learned features"]
        x_ticks = [3, 8]
        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0,
                   fontweight="bold", fontsize=14)
        plt.yticks(fontweight="bold", fontsize=14)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(model_name, fontweight="bold", fontsize=20)

        # legend
        legend_elements = []
        for i in range(len(pattern_names)):
            element = mpatches.Patch(color=colors[i], label=pattern_names[i])
            legend_elements.append(element)
        plt.legend(handles=legend_elements, prop={'size': 15},
                   loc='lower left', framealpha=1)

        # format plot
        plt.tight_layout()

        # output
        path = os.path.join(
            plot_directory, "auc_roc_{0}.png".format(model_name))
        plot.save_plot(path, "png")
        path = os.path.join(
            plot_directory, "auc_roc_{0}.pdf".format(model_name))
        plot.save_plot(path, "pdf")

        # show plot
        if show:
            plt.show()
        else:
            plt.close()

    return


def plot_dendrograms(
        learned_features,
        manual_features,
        df,
        plot_directory=None,
        show=False):
    # collect genes
    genes = list(set(df.loc[:, "gene"]))

    # define a ground truth cluster for each pattern
    pattern_to_cluster = {
        "random": 0,
        "foci": 1,
        "intranuclear": 2,
        "nuclear_edge": 3,
        "perinuclear": 4,
        "cell_edge": 5,
        "protrusion": 6}

    # loop over genes
    manual_features_genes = []
    learned_features_genes = []
    color_genes = []
    cluster_genes = []
    for gene in genes:

        # get mask gene
        mask_nopuromycin = df.loc[:, "puromycin"] == 0
        mask_gene = df.loc[:, "gene"] == gene
        mask = mask_nopuromycin & mask_gene

        # get manual features gene
        manual_features_gene = np.mean(manual_features[mask], axis=0)
        manual_features_genes.append(manual_features_gene)

        # get learned features gene
        learned_features_gene = np.mean(learned_features[mask], axis=0)
        learned_features_genes.append(learned_features_gene)

        # get pattern and color
        pattern, color = _assign_pattern_color(gene)
        cluster_gene = pattern_to_cluster[pattern]
        cluster_genes.append(cluster_gene)
        color_genes.append(color)

    # format embeddings and ground truth clusters
    manual_features_genes = np.stack(manual_features_genes)
    learned_features_genes = np.stack(learned_features_genes)
    cluster_genes = np.array(cluster_genes)

    # bugfix
    genes[genes.index("MYSNE2")] = "SYNE2"
    df_embeddings_manual = pd.DataFrame(manual_features_genes, index=genes)
    df_embeddings_learned = pd.DataFrame(learned_features_genes, index=genes)

    # compute linkage (manual)
    df_corr = df_embeddings_manual.T.corr()
    df_dism = 1 - np.abs(df_corr)
    linkage_manual = linkage(
        squareform(df_dism),
        method="average",
        metric="euclidean",
        optimal_ordering=True)

    # plot manual features (label)
    ax = sns.clustermap(
        data=df_dism,
        figsize=(10, 10),
        cbar_kws={"orientation": "horizontal"},
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage_manual,
        col_linkage=linkage_manual,
        row_colors=color_genes,
        col_colors=None,
        mask=None,
        dendrogram_ratio=0.2,
        colors_ratio=0.03,
        cbar_pos=(0.23, 0.84, 0.67, 0.03),
        tree_kws={"linewidths": 1.5})
    ax.ax_heatmap.set_xticks([])
    ax.ax_heatmap.set_yticklabels(
        labels=ax.ax_heatmap.get_yticklabels(),
        fontweight="bold", fontsize=10)
    ax.ax_col_dendrogram.set_visible(False)
    if plot_directory is not None:
        path = os.path.join(plot_directory, "dendrogram_manual_label.png")
        plot.save_plot(path, "png")
        path = os.path.join(plot_directory, "dendrogram_manual_label.pdf")
        plot.save_plot(path, "pdf")
    if show:
        plt.show()
    else:
        plt.close()

    # plot manual features (no label)
    ax = sns.clustermap(
        data=df_dism,
        figsize=(10, 10),
        cbar_kws={"orientation": "horizontal"},
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage_manual,
        col_linkage=linkage_manual,
        row_colors=None,
        col_colors=None,
        mask=None,
        dendrogram_ratio=0.2,
        colors_ratio=0.03,
        cbar_pos=(0.2, 0.84, 0.7, 0.03),
        tree_kws={"linewidths": 1.5})
    ax.ax_heatmap.set_xticks([])
    ax.ax_heatmap.set_yticklabels(
        labels=ax.ax_heatmap.get_yticklabels(),
        fontweight="bold", fontsize=10)
    ax.ax_col_dendrogram.set_visible(False)
    if plot_directory is not None:
        path = os.path.join(plot_directory, "dendrogram_manual_nolabel.png")
        plot.save_plot(path, "png")
        path = os.path.join(plot_directory, "dendrogram_manual_nolabel.pdf")
        plot.save_plot(path, "pdf")
    if show:
        plt.show()
    else:
        plt.close()

    # compute linkage (manual)
    df_corr = df_embeddings_learned.T.corr()
    df_dism = 1 - np.abs(df_corr)
    linkage_learned = linkage(
        squareform(df_dism),
        method='average',
        metric='euclidean',
        optimal_ordering=True)

    # plot learned features (label)
    ax = sns.clustermap(
        data=df_dism,
        figsize=(10, 10),
        cbar_kws={"orientation": "horizontal"},
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage_learned,
        col_linkage=linkage_learned,
        row_colors=color_genes,
        col_colors=None,
        mask=None,
        dendrogram_ratio=0.2,
        colors_ratio=0.03,
        cbar_pos=(0.23, 0.84, 0.67, 0.03),
        tree_kws={"linewidths": 1.5})
    ax.ax_heatmap.set_xticks([])
    ax.ax_heatmap.set_yticklabels(
        labels=ax.ax_heatmap.get_yticklabels(),
        fontweight="bold", fontsize=10)
    ax.ax_col_dendrogram.set_visible(False)
    if plot_directory is not None:
        path = os.path.join(plot_directory, "dendrogram_learned_label.png")
        plot.save_plot(path, "png")
        path = os.path.join(plot_directory, "dendrogram_learned_label.pdf")
        plot.save_plot(path, "pdf")
    if show:
        plt.show()
    else:
        plt.close()

    # plot learned features (no label)
    ax = sns.clustermap(
        data=df_dism,
        figsize=(10, 10),
        cbar_kws={"orientation": "horizontal"},
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage_learned,
        col_linkage=linkage_learned,
        row_colors=None,
        col_colors=None,
        mask=None,
        dendrogram_ratio=0.2,
        colors_ratio=0.03,
        cbar_pos=(0.2, 0.84, 0.7, 0.03),
        tree_kws={"linewidths": 1.5})
    ax.ax_heatmap.set_xticks([])
    ax.ax_heatmap.set_yticklabels(
        labels=ax.ax_heatmap.get_yticklabels(),
        fontweight="bold", fontsize=10)
    ax.ax_col_dendrogram.set_visible(False)
    if plot_directory is not None:
        path = os.path.join(plot_directory, "dendrogram_learned_nolabel.png")
        plot.save_plot(path, "png")
        path = os.path.join(plot_directory, "dendrogram_learned_nolabel.pdf")
        plot.save_plot(path, "pdf")
    if show:
        plt.show()
    else:
        plt.close()

    # compute Adjusted Rand Index (ARI)
    fc_manual = fcluster(linkage_manual, t=7, criterion="maxclust")
    rand_score_manual = adjusted_rand_score(cluster_genes, fc_manual)
    fc_learned = fcluster(linkage_learned, t=7, criterion="maxclust")
    rand_score_learned = adjusted_rand_score(cluster_genes, fc_learned)

    return rand_score_learned, rand_score_manual


def _assign_pattern_color(gene):
    pattern, color = None, None
    if gene in ["KIF20B", "MYO18A", "MYSNE2", "SYNE2", "PLEC"]:
        pattern = "random"
        color = "#a65628"
    elif gene in ["HMMR", "CEP170P1", "CTNNB1", "AURKA", "CRKL",
                  "DYNC1H1", "BUB1", "PAK2"]:
        pattern = "foci"
        color = "#ff7f00"
    elif gene in ["MYH3", "CEP192"]:
        pattern = "intranuclear"
        color = "#e41a1c"
    elif gene in ["ASPM", "SPEN"]:
        pattern = "nuclear_edge"
        color = "#377eb8"
    elif gene in ["AP1S2", "ATP6A2", "AKAP9", "AKAP1", "HSP90B1"]:
        pattern = "perinuclear"
        color = "#4daf4a"
    elif gene in ["FLNA"]:
        pattern = "cell_edge"
        color = "#000000"
    elif gene in ["KIF1C", "KIF4A", "RAB13", "DYNLL2", "KIF5B"]:
        pattern = "protrusion"
        color = "#984ea3"

    return pattern, color
