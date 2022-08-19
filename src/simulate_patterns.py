# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Simulate patterns.
"""

import os
import argparse

from utils import check_directories, initialize_script, end_script
from utils import plot_cell_coordinates

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import bigfish.stack as stack
import simfish as sim


# parameters
nb_simulations = 20000


def fct_to_parallelize(filename, template_directory, index_template,
                       simulation_directory, plot_directory):
    # get parameters
    parameters = filename.split("-")
    pattern = parameters[0]
    id_template = int(parameters[2])
    n_spots = int(parameters[3])
    proportion_pattern = int(parameters[4]) / 100

    # simulate coordinates
    simulated_coord = sim.simulate_localization_pattern(
        template_directory,
        n_spots=n_spots,
        i_cell=id_template,
        index_template=index_template,
        pattern=pattern,
        proportion_pattern=proportion_pattern)

    # save coordinates
    path = os.path.join(simulation_directory, "{0}.npz".format(filename))
    stack.save_cell_extracted(simulated_coord, path)

    # plot point cloud
    path = os.path.join(plot_directory, "{0}.png".format(filename))
    marge = stack.get_margin_value()
    cell_coord = simulated_coord["cell_coord"] + marge
    nuc_coord = simulated_coord["nuc_coord"] + marge
    rna_coord = simulated_coord["rna_coord"] + marge
    plot_cell_coordinates(
        ndim=3,
        cell_coord=cell_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord,
        framesize=(4, 4),
        path_output=path,
        show=False)

    return


if __name__ == "__main__":
    print()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory",
                        help="Path of the data directory.",
                        type=str)
    parser.add_argument("pattern",
                        help="Name of the pattern to simulate.",
                        type=str)
    parser.add_argument("log_directory",
                        help="Path of the log directory.",
                        type=str)

    # parse arguments
    args = parser.parse_args()
    data_directory = args.data_directory
    pattern = args.pattern
    log_directory = args.log_directory

    # initialize parameters
    template_directory = os.path.join(data_directory, "templates")
    simulation_directory = os.path.join(data_directory, "simulations")
    plot_directory = os.path.join(data_directory, "plots")

    # check directories exists
    check_directories([data_directory, log_directory, template_directory,
                       simulation_directory, plot_directory])

    # initialize script
    start_time, path_log_directory = initialize_script(
        training_directory=log_directory, experiment_name=pattern)

    # get template index
    df_template = sim.read_index_template(template_directory)
    if pattern == "protrusion":
        mask = df_template.loc[:, "protrusion_flag"] == "protrusion"
        df_template = df_template.loc[mask, :]

    # get potential templates
    l_id_template = list(df_template.loc[:, "id"])
    l_id_template = np.random.choice(
        l_id_template, size=nb_simulations, replace=True)

    # get potential number of spots
    l_n_spots_ = np.logspace(
        np.log(50), np.log(900), nb_simulations, base=np.exp(1), dtype=np.int)
    probabilities, _ = np.histogram(l_n_spots_, bins=18, range=(49, 901))
    probabilities = probabilities / np.sum(probabilities)
    values = [i for i in range(50, 901, 50)]
    l_n_spots = np.random.choice(
        values, size=nb_simulations, replace=True, p=probabilities)

    # get potential proportion of pattern
    if pattern == "random":
        l_proportion_pattern = [0] * nb_simulations
    elif pattern in ["foci", "intranuclear", "extranuclear", "nuclear_edge",
                     "perinuclear", "cell_edge", "pericellular"]:
        l_proportion_pattern = np.random.randint(60, 101, nb_simulations)
    elif pattern == "protrusion":
        l_proportion_pattern = np.random.randint(20, 51, nb_simulations)
    else:
        l_proportion_pattern = None

    #  build filenames to simulate
    filenames = []
    for id_cell in range(nb_simulations):

        # get filename
        id_template = l_id_template[id_cell]
        n_spots = l_n_spots[id_cell]
        proportion_pattern = l_proportion_pattern[id_cell]
        filename = "{0}-{1}-{2}-{3}-{4}".format(
            pattern, id_cell, id_template, n_spots, proportion_pattern)
        filenames.append(filename)

    # create dataframe with metadata
    df = pd.DataFrame()
    df.loc[:, "filename"] = filenames
    df.loc[:, "id_template"] = l_id_template
    df.loc[:, "pattern"] = [pattern] * nb_simulations
    df.loc[:, "n_spots"] = l_n_spots
    df.loc[:, "proportion_pattern"] = l_proportion_pattern
    df.loc[:, "id_simulation"] = [i for i in range(nb_simulations)]

    # define function to simulate targeted pattern
    def fct_pattern(filename):
        return fct_to_parallelize(
            filename=filename,
            template_directory=template_directory,
            index_template=df_template,
            simulation_directory=simulation_directory,
            plot_directory=plot_directory)

    # parallelize
    Parallel(n_jobs=-1)(delayed(fct_pattern)(filename)
                        for filename in filenames)

    # save metadata dataframe
    path = os.path.join(data_directory, "{0}.csv".format(pattern))
    stack.save_data_to_csv(df, path)

    print()

    print("Dataframe shape: {0}".format(df.shape))
    end_script(start_time)
