# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""Routine functions."""

import os
import time
import datetime
import sys
import inspect

from distutils.dir_util import copy_tree

import numpy as np
import matplotlib.pyplot as plt

import bigfish.stack as stack
import bigfish.plot as plot


# ### Initialization ###

def check_directories(path_directories):
    # check directories exist
    stack.check_parameter(path_directories=list)
    for path_directory in path_directories:
        if not os.path.isdir(path_directory):
            raise ValueError("Directory does not exist: {0}"
                             .format(path_directory))

    return


def initialize_script(training_directory, experiment_name=None, array_id=None):
    # check parameters
    stack.check_parameter(training_directory=str,
                          experiment_name=(str, type(None)))

    # get filename of the script that call this function
    try:
        previous_filename = inspect.getframeinfo(sys._getframe(1))[0]
    except ValueError:
        previous_filename = None

    # get date of execution
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")
    year = date.split("-")[0]
    month = date.split("-")[1]
    day = date.split("-")[2].split(" ")[0]
    hour = date.split(":")[0].split(" ")[1]
    minute = date.split(":")[1]
    second = date.split(":")[2]

    # format log name
    log_date = year + month + day + hour + minute + second
    if previous_filename is not None:
        operation = os.path.basename(previous_filename)
        operation = operation.split(".")[0]
        if experiment_name is not None:
            log_name = "{0}_{1}_{2}".format(
                log_date, operation, experiment_name)
        else:
            log_name = "{0}_{1}".format(log_date, operation)
    else:
        if experiment_name is not None:
            log_name = "{0}_{1}".format(log_date, experiment_name)
        else:
            log_name = "{0}".format(log_date)
    if array_id is not None:
        log_name += "_{0}".format(array_id)

    # initialize logging in a specific log directory
    path_log_directory = os.path.join(training_directory, log_name)
    os.mkdir(path_log_directory)
    path_log_file = os.path.join(path_log_directory, "log.txt")
    sys.stdout = Logger(path_log_file)

    # copy python script in the log directory
    if previous_filename is not None:
        path_src = os.path.dirname(previous_filename)
        path_output = os.path.join(path_log_directory, "src")
        os.mkdir(path_output)
        copy_tree(path_src, path_output)

    # print information about launched script
    if previous_filename is not None:
        print("Running {0} file..."
              .format(os.path.basename(previous_filename)))
        print()
    start_time = time.time()
    if experiment_name is not None:
        print("Experiment name: {0}".format(experiment_name))
    print("Log directory: {0}".format(path_log_directory))
    print("Log name: {0}".format(log_name))
    print("Date: {0}".format(date), "\n")

    return start_time, path_log_directory


def end_script(start_time):
    # check parameters
    stack.check_parameter(start_time=(int, float))

    # time the script
    end_time = time.time()
    duration = int(round((end_time - start_time) / 60))
    print("Duration: {0} minutes".format(duration))
    print("-------------------------", "\n\n\n")

    return


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# ### Plot ###

def plot_cell_coordinates(ndim, cell_coord, nuc_coord, rna_coord, titles=None,
                          remove_frame=True, framesize=(10, 5),
                          path_output=None, ext="png", show=True):
    """
    Plot cell coordinates for one or several cells.

    Parameters
    ----------
    ndim : {2, 3}
        Number of spatial dimensions to consider in the coordinates.
    cell_coord : np.ndarray or list
        Coordinates or list of coordinates of the cell border with shape
        (nb_points, 2).
    nuc_coord : np.ndarray or list
        Coordinates or list of coordinates of the nucleus border with shape
        (nb_points, 2).
    rna_coord : np.ndarray or list
        Coordinates or list of coordinates of the detected spots with shape
        (nb_spots, 3) or (nb_spots, 2). One coordinate per dimension (zyx or
        yx dimensions).
    titles : str or list, optional
        Title or list of titles.
    remove_frame : bool, default=True
        Remove axes and frame.
    framesize : tuple, default=(10, 5)
        Size of the frame.
    path_output : str, optional
        Path to save the image (without extension).
    ext : str or list, default='png'
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool, default=True
        Show the figure or not.

    """
    # enlist coordinates if necessary
    if isinstance(cell_coord, np.ndarray):
        cell_coord = [cell_coord]
    if isinstance(nuc_coord, np.ndarray):
        nuc_coord = [nuc_coord]
    if isinstance(rna_coord, np.ndarray):
        rna_coord = [rna_coord]

    # check parameters
    stack.check_parameter(
        ndim=int,
        titles=(str, list, type(None)),
        remove_frame=bool,
        framesize=tuple,
        path_output=(str, type(None)),
        ext=(str, list))

    # check coordinates
    for i in range(len(cell_coord)):
        stack.check_array(cell_coord[i], ndim=2, dtype=[np.int64, np.float64])
        stack.check_array(nuc_coord[i], ndim=2, dtype=[np.int64, np.float64])
        stack.check_array(rna_coord[i], ndim=2, dtype=[np.int64, np.float64])

    # enlist 'titles' if needed
    if titles is not None and isinstance(titles, str):
        titles = [titles]

    # we plot 3 images by row maximum
    nrow = int(np.ceil(len(cell_coord) / 3))
    ncol = min(len(cell_coord), 3)

    # plot one image
    if len(cell_coord) == 1:

        # frame
        if remove_frame:
            fig = plt.figure(figsize=framesize, frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
        else:
            plt.figure(figsize=framesize)

        # coordinate image
        plt.plot(
            cell_coord[0][:, 1],
            cell_coord[0][:, 0],
            c="black",
            linewidth=2)
        plt.plot(
            nuc_coord[0][:, 1],
            nuc_coord[0][:, 0],
            c="steelblue",
            linewidth=2)
        plt.scatter(
            rna_coord[0][:, ndim - 1],
            rna_coord[0][:, ndim - 2],
            s=25,
            c="firebrick",
            marker=".")

        # titles and frames
        _, _, min_y, max_y = plt.axis()
        plt.ylim(max_y, min_y)
        plt.use_sticky_edges = True
        plt.margins(0.01, 0.01)
        plt.axis('scaled')
        if titles is not None:
            plt.title(
                titles[0],
                fontweight="bold",
                fontsize=10)
        if not remove_frame:
            plt.tight_layout()

        # output
        if path_output is not None:
            plot.save_plot(path_output, ext)
        if show:
            plt.show()
        else:
            plt.close()

        return

    # plot multiple images
    fig, ax = plt.subplots(nrow, ncol, figsize=framesize)

    # one row
    if len(cell_coord) in [2, 3]:

        # loop over instance coordinates
        for i in range(len(cell_coord)):

            # coordinate image
            ax[i].plot(
                cell_coord[i][:, 1],
                cell_coord[i][:, 0],
                c="black",
                linewidth=2)
            ax[i].plot(
                nuc_coord[i][:, 1],
                nuc_coord[i][:, 0],
                c="steelblue",
                linewidth=2)
            ax[i].scatter(
                rna_coord[i][:, ndim - 1],
                rna_coord[i][:, ndim - 2],
                s=25,
                c="firebrick",
                marker=".")

            # titles and frames
            _, _, min_y, max_y = ax[i].axis()
            ax[i].set_ylim(max_y, min_y)
            ax[i].use_sticky_edges = True
            ax[i].margins(0.01, 0.01)
            ax[i].axis('scaled')
            if remove_frame:
                ax[i].axis("off")
            if titles is not None:
                ax[i].set_title(
                    titles[i],
                    fontweight="bold",
                    fontsize=10)

    # several rows
    else:

        # we complete the row with empty frames
        r = nrow * 3 - len(cell_coord)
        cell_coord_completed = [cell_coord_ for cell_coord_ in cell_coord]
        cell_coord_completed += [None] * r

        # loop over instance coordinates
        for i in range(len(cell_coord_completed)):
            row = i // 3
            col = i % 3

            # empty subplot
            if cell_coord_completed[i] is None:
                ax[row, col].set_visible(False)
                continue

            # coordinate image
            ax[row, col].plot(
                cell_coord_completed[i][:, 1],
                cell_coord_completed[i][:, 0],
                c="black",
                linewidth=2)
            ax[row, col].plot(
                nuc_coord[i][:, 1],
                nuc_coord[i][:, 0],
                c="steelblue",
                linewidth=2)
            ax[row, col].scatter(
                rna_coord[i][:, ndim - 1],
                rna_coord[i][:, ndim - 2],
                s=25,
                c="firebrick",
                marker=".")

            # titles and frames
            _, _, min_y, max_y = ax[row, col].axis()
            ax[row, col].set_ylim(max_y, min_y)
            ax[row, col].use_sticky_edges = True
            ax[row, col].margins(0.01, 0.01)
            ax[row, col].axis('scaled')
            if remove_frame:
                ax[row, col].axis("off")
            if titles is not None:
                ax[row, col].set_title(
                    titles[i],
                    fontweight="bold",
                    fontsize=10)

    # output
    plt.tight_layout()
    if path_output is not None:
        plot.save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()
