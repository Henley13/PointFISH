# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Build tfrecord files to feed a model.
"""

import os
import argparse
import shutil

import numpy as np
import pandas as pd

import bigfish.stack as stack
import simfish as sim

from joblib import Parallel
from joblib import delayed

from utils import check_directories
from utils import initialize_script
from utils import end_script

from preprocessing import extract_internal_boundary_coord
from preprocessing import build_example_clf
from preprocessing import write_tfrecord


def merge_df(directory_in, directory_out, patterns):
    # read dataframes
    dataframes = []
    for pattern in patterns:
        path = os.path.join(directory_in, "{0}.csv".format(pattern))
        df = stack.read_dataframe_from_csv(path)
        dataframes.append(df)

    # merge dataframes
    df_all = pd.concat(dataframes)
    df_all.reset_index(drop=True, inplace=True)

    # save dataframe
    path = os.path.join(directory_out, "all_patterns.csv")
    stack.save_data_to_csv(df_all, path)

    return df_all


def split_simulations_templates(template_directory, p_train=0.6, p_val=0.2):
    # get cell id with and without protrusion
    df_template = sim.read_index_template(template_directory)
    mask = df_template.loc[:, "protrusion_flag"] == "protrusion"
    df_protrusion = df_template.loc[mask, :]
    id_template_protrusion = list(set(df_protrusion.loc[:, "id"]))
    mask = df_template.loc[:, "protrusion_flag"] == "noprotrusion"
    df_noprotrusion = df_template.loc[mask, :]
    id_template_noprotrusion = list(set(df_noprotrusion.loc[:, "id"]))
    id_template = id_template_protrusion + id_template_noprotrusion

    # shuffle
    np.random.shuffle(id_template_protrusion)
    np.random.shuffle(id_template_noprotrusion)

    # split train/validation/test (protrusion)
    nb_train = int(p_train * len(id_template_protrusion))
    nb_val = int(p_val * len(id_template_protrusion))
    train_template_protrusion = id_template_protrusion[:nb_train]
    val_template_protrusion = id_template_protrusion[nb_train:nb_train+nb_val]

    # split train/validation/test (no protrusion)
    nb_train = int(p_train * len(id_template_noprotrusion))
    nb_val = int(p_val * len(id_template_noprotrusion))
    train_template_noprotrusion = id_template_noprotrusion[:nb_train]
    val_template_noprotrusion = id_template_noprotrusion[nb_train:nb_train+nb_val]

    # split train/validation/test
    train_template = train_template_protrusion + train_template_noprotrusion
    train_template = list(train_template)
    val_template = val_template_protrusion + val_template_noprotrusion
    val_template = list(val_template)
    test_template = set(id_template) - set(train_template) - set(val_template)
    test_template = list(test_template)

    return train_template, val_template, test_template


def from_point_cloud_to_tfrecord_clf(
        rna_coord,
        cell_coord,
        nuc_coord,
        cell_mask,
        nuc_mask,
        pattern,
        add_cluster,
        voxel_size,
        add_morphology,
        n_coord_cell,
        n_coord_nuc,
        add_distance,
        normalized,
        random_rotation,
        sample_name):
    # cast coordinates
    rna_coord = np.round(rna_coord, 0).astype(np.int64)
    cell_coord = np.round(cell_coord, 0).astype(np.int64)
    nuc_coord = np.round(nuc_coord, 0).astype(np.int64)

    # get example
    example = build_example_clf(
        rna_coord=rna_coord,
        cell_coord=cell_coord,
        nuc_coord=nuc_coord,
        cell_mask=cell_mask,
        nuc_mask=nuc_mask,
        pattern=pattern,
        add_cluster=add_cluster,
        voxel_size=voxel_size,
        add_morphology=add_morphology,
        n_coord_cell=n_coord_cell,
        n_coord_nuc=n_coord_nuc,
        add_distance=add_distance,
        normalized=normalized,
        random_rotation=random_rotation,
        name=sample_name)

    return example


def get_real_metadata(df):
    # parameters
    filenames = []
    patterns = []
    annotated_patterns_flag = [
        "pattern_foci", "pattern_intranuclear", "pattern_nuclear",
        "pattern_perinuclear", "pattern_cell", "pattern_protrusion"]
    annotated_patterns = [
        "foci", "intranuclear", "nuclear_edge", "perinuclear", "cell_edge",
        "protrusion"]

    # loop over real instances
    for i, row in df.iterrows():

        # get filenames
        filename = row["cell"]
        filenames.append(filename)

        # get patterns
        annotated = row["annotated"]
        if not annotated:
            patterns.append(None)
        else:
            patterns_ = []
            for pattern_flag, pattern in zip(annotated_patterns_flag,
                                             annotated_patterns):
                if row[pattern_flag]:
                    patterns_.append(pattern)
            if len(patterns_) == 0:
                patterns_.append("random")
            patterns.append(patterns_)

    return filenames, patterns


if __name__ == "__main__":
    print()

    # get number of CPUs
    nb_cpu = os.cpu_count()
    print("Number of CPUs: {0}".format(nb_cpu))
    print()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory",
                        help="Path of the data directory.",
                        type=str)
    parser.add_argument("log_directory",
                        help="Path of the log directory.",
                        type=str)

    # parse arguments
    args = parser.parse_args()
    data_directory = args.data_directory
    log_directory = args.log_directory

    # paths
    output_directory = os.path.join(data_directory, "data_clf")
    template_directory = os.path.join(data_directory, "templates")
    simulation_directory = os.path.join(data_directory, "simulations")
    real_directory = os.path.join(data_directory, "reals")

    # check directories exists
    check_directories([
        data_directory,
        output_directory,
        template_directory,
        simulation_directory,
        real_directory,
        log_directory])

    # build tfrecords directories (simulation)
    tfrecord_directory_clf = os.path.join(
        output_directory, "tfrecords_clf")
    tfrecord_directory_morphology_clf = os.path.join(
        output_directory, "tfrecords_morphology_clf")
    tfrecords_directories = [
        tfrecord_directory_clf,
        tfrecord_directory_morphology_clf]
    for d in tfrecords_directories:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.mkdir(d)
        path = os.path.join(d, "train")
        os.mkdir(path)
        path = os.path.join(d, "validation")
        os.mkdir(path)
        path = os.path.join(d, "test")
        os.mkdir(path)

    # initialize script
    start_time, path_log_directory = initialize_script(log_directory)

    print("--- Build tfrecords classification --- \n")

    # collect and merge metadata
    patterns = ["random", "foci", "intranuclear", "extranuclear",
                "nuclear_edge", "perinuclear", "cell_edge", "pericellular"]
    df_all = merge_df(data_directory, output_directory, patterns=patterns)

    # split simulation templates
    train_template, val_template, test_template = split_simulations_templates(
        template_directory, p_train=0.6, p_val=0.2)

    # build train/validation/test filenames
    query = "id_template in {0}".format(train_template)
    df_all_train = df_all.query(query)
    train_filenames = list(df_all_train.loc[:, "filename"])
    query = "id_template in {0}".format(val_template)
    df_all_val = df_all.query(query)
    val_filenames = list(df_all_val.loc[:, "filename"])
    query = "id_template in {0}".format(test_template)
    df_all_test = df_all.query(query)
    test_filenames = list(df_all_test.loc[:, "filename"])
    d_filenames = {
        "train": train_filenames,
        "validation": val_filenames,
        "test": test_filenames}

    # save dataframes split
    path = os.path.join(output_directory, "all_patterns_train.csv")
    stack.save_data_to_csv(df_all_train, path)
    path = os.path.join(output_directory, "all_patterns_validation.csv")
    stack.save_data_to_csv(df_all_val, path)
    path = os.path.join(output_directory, "all_patterns_test.csv")
    stack.save_data_to_csv(df_all_test, path)

    # define function to distribute across CPUs
    def fct_to_process(filename):

        # read data
        path = os.path.join(simulation_directory, "{0}.npz".format(filename))
        data = stack.read_cell_extracted(path)
        rna_coord = data["rna_coord"]
        cell_mask = data["cell_mask"]
        nuc_mask = data["nuc_mask"]

        # get cell and nucleus coordinates (internal boundaries)
        cell_coord = extract_internal_boundary_coord(cell_mask)
        nuc_coord = extract_internal_boundary_coord(nuc_mask)

        # get metadata
        filename_split = filename.split("-")
        pattern = filename_split[0]

        # check number of RNAs
        nb_rna_template = [i for i in range(50, 901, 50)]
        if len(rna_coord) not in nb_rna_template:
            return None

        # write tfrecords classification (RNA only)
        example_1 = from_point_cloud_to_tfrecord_clf(
            rna_coord=rna_coord,
            cell_coord=cell_coord,
            nuc_coord=nuc_coord,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            pattern=pattern,
            add_cluster=True,
            voxel_size=(100, 100, 100),
            add_morphology=False,
            n_coord_cell=300,
            n_coord_nuc=100,
            add_distance=True,
            normalized=True,
            random_rotation=True,
            sample_name=filename)

        # write tfrecords classification (RNA + morphology 2D)
        example_2 = from_point_cloud_to_tfrecord_clf(
            rna_coord=rna_coord,
            cell_coord=cell_coord,
            nuc_coord=nuc_coord,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            pattern=pattern,
            add_cluster=True,
            voxel_size=(100, 100, 100),
            add_morphology=True,
            n_coord_cell=300,
            n_coord_nuc=100,
            add_distance=True,
            normalized=True,
            random_rotation=True,
            sample_name=filename)

        return example_1, example_2

    # define parallel workers
    with Parallel(n_jobs=-1) as parallel:

        # loop over splits
        nb_tfrecords = {
            "train": 0,
            "validation": 0,
            "test": 0}
        for split in ["train", "validation", "test"]:

            filenames = d_filenames[split]
            np.random.shuffle(filenames)
            print("Number of {0} simulations: {1}".format(
                split, len(filenames)))

            # save a new tfrecords file every 1000 observations
            n_start = 0
            n_end = n_start + 1000
            while n_start < len(filenames):
                n_end = min(n_end, len(filenames))
                sample_filenames = filenames[n_start: n_end]

                # loop over filenames
                examples = parallel(delayed(fct_to_process)(filename)
                                    for filename in sample_filenames)
                examples = [x for x in examples if x is not None]
                nb_tfrecords[split] += len(examples)
                examples_1, examples_2 = zip(*examples)
                examples_1 = list(examples_1)
                examples_2 = list(examples_2)

                # write larger tfrecords file
                path = os.path.join(
                    tfrecord_directory_clf, split,
                    "data_{0}_{1}.tfrecords".format(n_start, n_end))
                write_tfrecord(examples_1, path)
                path = os.path.join(
                    tfrecord_directory_morphology_clf, split,
                    "data_{0}_{1}.tfrecords".format(n_start, n_end))
                write_tfrecord(examples_2, path)

                # update indices
                n_start = n_end
                n_end += 1000

    print("Number of train tfrecords: {0}".format(
        nb_tfrecords["train"]))
    print("Number of validation tfrecords: {0}".format(
        nb_tfrecords["validation"]))
    print("Number of test tfrecords: {0}".format(
        nb_tfrecords["test"]))
    print("Total number of tfrecords: {0}".format(
        nb_tfrecords["train"]
        + nb_tfrecords["validation"]
        + nb_tfrecords["test"]))
    print()

    end_script(start_time)
