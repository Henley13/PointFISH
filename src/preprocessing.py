# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Prepare data to feed a model.
"""

import os
import tensorflow as tf
import numpy as np
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.segmentation as segmentation
import bigfish.multistack as multistack

from scipy import ndimage as ndi
from skimage.measure import find_contours


# ### Build features ###

def extract_internal_boundary_coord(binary_surface, smooth_mask=False,
                                    smoothness=5):
    # smooth binary mask
    if smooth_mask:
        binary_surface = segmentation.clean_segmentation(
            image=binary_surface,
            small_object_size=100,
            smoothness=smoothness)

    # format surface mask
    binary_boundary = multistack.from_surface_to_boundaries(binary_surface)
    mask_surface = binary_surface.copy().astype(np.uint8)
    mask_surface *= 2
    mask_surface[binary_boundary] = 1

    # extract boundaries coordinates
    mask_surface = np.pad(mask_surface, [(1, 1)], mode="constant")
    coord = find_contours(
        mask_surface,
        level=1,
        fully_connected='low',
        positive_orientation='low')[0].astype(np.int64)
    coord -= 1

    # complete coordinates
    coord = multistack.complete_coord_boundaries(coord)

    return coord


def get_real_binary_mask(cell_label, nuc_label):
    # get center coordinate
    y_center = int(cell_label.shape[0] / 2)
    x_center = int(cell_label.shape[1] / 2)

    # get cell and nucleus id
    i_instance = cell_label[y_center, x_center]

    # get binary masks
    cell_mask = cell_label == i_instance
    nuc_mask = nuc_label == i_instance

    return cell_mask, nuc_mask


def morphology_coord(rna_coord, cell_coord, nuc_coord):
    ndim = rna_coord.shape[1]
    if ndim == 3:
        z_centroid = rna_coord.mean(axis=0)[0]
        extra_dim_cell = np.array([z_centroid] * cell_coord.shape[0])
        extra_dim_cell = extra_dim_cell[:, np.newaxis]
        new_cell_coord = np.hstack(
            [extra_dim_cell, cell_coord]).astype(np.float32)
        extra_dim_nuc = np.array([z_centroid] * nuc_coord.shape[0])
        extra_dim_nuc = extra_dim_nuc[:, np.newaxis]
        new_nuc_coord = np.hstack(
            [extra_dim_nuc, nuc_coord]).astype(np.float32)
    else:
        new_cell_coord = cell_coord.copy().astype(np.float32)
        new_nuc_coord = nuc_coord.copy().astype(np.float32)

    return new_cell_coord, new_nuc_coord


def distance_from_cell(coord, cell_mask, normalization):
    # compute distance map
    distance_cell = ndi.distance_transform_edt(cell_mask)
    distance_cell = distance_cell.astype(np.float32)

    # normalize according to cell morphology
    if normalization:
        m = distance_cell.ravel().max()
        distance_cell /= m

    # extract distance from cell membrane
    object_distance_cell = distance_cell[coord[:, 0], coord[:, 1]]

    return object_distance_cell


def distance_from_nuc(coord, cell_mask, nuc_mask, normalization):
    # compute distance map
    distance_nuc_out = ndi.distance_transform_edt(~nuc_mask)
    distance_nuc_out = cell_mask * distance_nuc_out
    distance_nuc_out = distance_nuc_out.astype(np.float32)
    distance_nuc_in = ndi.distance_transform_edt(nuc_mask)
    distance_nuc_in = distance_nuc_in.astype(np.float32)
    distance_nuc = distance_nuc_out - distance_nuc_in

    # normalize according to cell morphology
    if normalization:
        m = np.abs(distance_nuc.ravel()).max()
        distance_nuc /= m

    # extract distance from nucleus membrane
    object_distance_nuc = distance_nuc[coord[:, 0], coord[:, 1]]

    return object_distance_nuc


def normalize_coord(coord):
    normalized_coord = coord.copy().astype(np.float32)
    centroid = np.mean(coord, axis=0)
    normalized_coord -= centroid
    max_distance = np.max(np.sqrt(np.sum(normalized_coord ** 2, axis=-1)))
    normalized_coord /= max_distance

    return normalized_coord


def build_features(
        rna_coord,
        cell_coord,
        nuc_coord,
        cell_mask,
        nuc_mask,
        add_cluster=False,
        voxel_size=(100, 100, 100),
        add_morphology=False,
        n_coord_cell=300,
        n_coord_nuc=100,
        add_distance=False,
        normalized=True,
        random_rotation=False):
    # get coordinate dimension
    ndim = rna_coord.shape[1]
    if len(voxel_size) != ndim:
        raise ValueError("'voxel_size' should have {0} elements, not {1}"
                         .format(ndim, len(voxel_size)))

    # add cell-nucleus coordinates
    if add_morphology:
        (features_coord,
         features_cluster,
         features_morphology,
         features_distance) = _build_features_morphology(
            rna_coord=rna_coord,
            cell_coord=cell_coord,
            nuc_coord=nuc_coord,
            ndim=ndim,
            voxel_size=voxel_size,
            normalized=normalized,
            random_rotation=random_rotation,
            add_cluster=add_cluster,
            n_coord_cell=n_coord_cell,
            n_coord_nuc=n_coord_nuc,
            add_distance=add_distance,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask)

    else:
        (features_coord,
         features_cluster,
         features_morphology,
         features_distance) = _build_features_nomorphology(
            rna_coord=rna_coord,
            cell_coord=cell_coord,
            nuc_coord=nuc_coord,
            ndim=ndim,
            voxel_size=voxel_size,
            normalized=normalized,
            random_rotation=random_rotation,
            add_cluster=add_cluster,
            add_distance=add_distance,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask)

    return (features_coord,
            features_cluster,
            features_morphology,
            features_distance)


def _build_features_morphology(
        rna_coord,
        cell_coord,
        nuc_coord,
        ndim,
        voxel_size,
        normalized,
        random_rotation,
        add_cluster,
        n_coord_cell,
        n_coord_nuc,
        add_distance,
        cell_mask,
        nuc_mask):
    # ### coordinates features ###

    # random rotation
    if random_rotation:
        angle = np.random.randint(0, 361)
        features_coord_cell, features_coord_nuc, features_coord_rna = rotation(
            cell_coord=cell_coord,
            nuc_coord=nuc_coord,
            rna_coord=rna_coord,
            angle=angle,
            ndim=ndim)
    else:
        features_coord_cell = cell_coord.copy()
        features_coord_nuc = nuc_coord.copy()
        features_coord_rna = rna_coord.copy()

    # get cell-nucleus coordinates
    features_coord_cell, features_coord_nuc = morphology_coord(
        features_coord_rna, features_coord_cell, features_coord_nuc)

    # sample cell-nucleus coordinates
    indices = np.arange(len(features_coord_cell))
    sample_indices_cell = np.random.choice(
        indices, n_coord_cell, replace=False)
    sample_indices_cell = np.sort(sample_indices_cell)
    features_coord_cell = features_coord_cell[sample_indices_cell]
    indices = np.arange(len(features_coord_nuc))
    sample_indices_nuc = np.random.choice(
        indices, n_coord_nuc, replace=False)
    sample_indices_nuc = np.sort(sample_indices_nuc)
    features_coord_nuc = features_coord_nuc[sample_indices_nuc]

    # stack features coordinates
    features_coord = np.concatenate(
        [features_coord_rna.astype(np.float32),
         features_coord_cell,
         features_coord_nuc])

    # convert coordinates in nanometers
    # TODO remove casting if big-fish 0.6.2
    features_coord = detection.convert_spot_coordinates(
        spots=features_coord.astype(np.float64),
        voxel_size=voxel_size)

    # normalized coordinates
    if normalized:
        features_coord = normalize_coord(features_coord)
    else:
        features_coord = features_coord.astype(np.float32)

    # ### cluster features ###

    # add cluster flag
    if add_cluster:
        rna_coord_clustered, _ = detection.detect_clusters(
            spots=rna_coord,
            voxel_size=voxel_size,
            radius=350,
            nb_min_spots=5)
        features_cluster_rna = rna_coord_clustered[:, ndim] > -1
        features_cluster_cell = np.array([0] * n_coord_cell)
        features_cluster_nuc = np.array([0] * n_coord_nuc)
        features_cluster = np.concatenate(
            [features_cluster_rna.astype(np.float32),
             features_cluster_cell.astype(np.float32),
             features_cluster_nuc.astype(np.float32)])
        features_cluster = features_cluster[:, np.newaxis]
    else:
        features_cluster = None

    # ### morphology features ###

    # add node type
    features_cell_type = ([0] * len(rna_coord)
                          + [1] * n_coord_cell
                          + [0] * n_coord_nuc)
    features_cell_type = np.array(features_cell_type).astype(np.float32)
    features_cell_type = features_cell_type[:, np.newaxis]
    features_nuc_type = ([0] * len(rna_coord)
                         + [0] * n_coord_cell
                         + [1] * n_coord_nuc)
    features_nuc_type = np.array(features_nuc_type).astype(np.float32)
    features_nuc_type = features_nuc_type[:, np.newaxis]
    features_morphology = np.hstack([features_cell_type, features_nuc_type])

    # ### distance features ###

    # add distances
    if add_distance:

        # distance from cell
        features_distance_rna_cell = distance_from_cell(
            coord=rna_coord[:, ndim - 2:].astype(np.int64),
            cell_mask=cell_mask,
            normalization=normalized)
        features_distance_cell_cell = np.array(
            [0] * n_coord_cell, dtype=np.float32)
        features_distance_nuc_cell = distance_from_cell(
            coord=nuc_coord,
            cell_mask=cell_mask,
            normalization=normalized)
        features_distance_nuc_cell = features_distance_nuc_cell[
            sample_indices_nuc]
        features_distance_cell = np.concatenate(
            [features_distance_rna_cell,
             features_distance_cell_cell,
             features_distance_nuc_cell])
        if not normalized:
            features_distance_cell *= voxel_size[-1]
        features_distance_cell = features_distance_cell[:, np.newaxis]

        # distance from nucleus
        features_distance_rna_nuc = distance_from_nuc(
            coord=rna_coord[:, ndim - 2:].astype(np.int64),
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            normalization=normalized)
        features_distance_cell_nuc = distance_from_nuc(
            coord=cell_coord,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            normalization=normalized)
        features_distance_cell_nuc = features_distance_cell_nuc[
            sample_indices_cell]
        features_distance_nuc_nuc = np.array(
            [0] * n_coord_nuc, dtype=np.float32)
        features_distance_nuc = np.concatenate(
            [features_distance_rna_nuc,
             features_distance_cell_nuc,
             features_distance_nuc_nuc])
        if not normalized:
            features_distance_nuc *= voxel_size[-1]
        features_distance_nuc = features_distance_nuc[:, np.newaxis]

        # stack distances
        features_distance = np.hstack(
            [features_distance_cell, features_distance_nuc])

    else:
        features_distance = None

    return (features_coord,
            features_cluster,
            features_morphology,
            features_distance)


def _build_features_nomorphology(
        rna_coord,
        cell_coord,
        nuc_coord,
        ndim,
        voxel_size,
        normalized,
        random_rotation,
        add_cluster,
        add_distance,
        cell_mask,
        nuc_mask):
    # ### coordinates features ###

    # random rotation
    if random_rotation:
        angle = np.random.randint(0, 361)
        _, _, features_coord = rotation(
            cell_coord=cell_coord,
            nuc_coord=nuc_coord,
            rna_coord=rna_coord,
            angle=angle,
            ndim=ndim)
    else:
        features_coord = rna_coord.copy()

    # convert coordinates in nanometers
    # TODO remove casting if big-fish 0.6.2
    features_coord = detection.convert_spot_coordinates(
        spots=features_coord.astype(np.float64),
        voxel_size=voxel_size)

    # normalized coordinates
    if normalized:
        features_coord = normalize_coord(features_coord)
    else:
        features_coord = features_coord.astype(np.float32)

    # ### cluster features ###

    # add cluster flag
    if add_cluster:
        rna_coord_clustered, _ = detection.detect_clusters(
            spots=rna_coord,
            voxel_size=voxel_size,
            radius=350,
            nb_min_spots=5)
        features_cluster = rna_coord_clustered[:, ndim] > -1
        features_cluster = features_cluster.astype(np.float32)
        features_cluster = features_cluster[:, np.newaxis]
    else:
        features_cluster = None

    # ### morphology features ###

    features_morphology = None

    # ### distance features ###

    # add distances
    if add_distance:

        # distance from cell
        features_distance_cell = distance_from_cell(
            coord=rna_coord[:, ndim - 2:].astype(np.int64),
            cell_mask=cell_mask,
            normalization=normalized)
        if not normalized:
            features_distance_cell *= voxel_size[-1]
        features_distance_cell = features_distance_cell[:, np.newaxis]

        # distance from nucleus
        features_distance_nuc = distance_from_nuc(
            coord=rna_coord[:, ndim - 2:].astype(np.int64),
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            normalization=normalized)
        if not normalized:
            features_distance_nuc *= voxel_size[-1]
        features_distance_nuc = features_distance_nuc[:, np.newaxis]

        # stack distances
        features_distance = np.hstack(
            [features_distance_cell, features_distance_nuc])

    else:
        features_distance = None

    return (features_coord,
            features_cluster,
            features_morphology,
            features_distance)


# ### Rotation ###

def cartesian_to_polar(coord):
    ndim = coord.shape[1]
    if ndim == 3:
        z, y, x = coord[:, 0], coord[:, 1], coord[:, 2]
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        new_coord = np.stack([z, r, phi], axis=-1)
    else:
        y, x = coord[:, 0], coord[:, 1]
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        new_coord = np.stack([r, phi], axis=-1)

    return new_coord


def polar_to_cartesian(coord):
    ndim = coord.shape[1]
    if ndim == 3:
        z, r, phi = coord[:, 0], coord[:, 1], coord[:, 2]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        new_coord = np.stack([z, y, x], axis=-1)
    else:
        r, phi = coord[:, 0], coord[:, 1]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        new_coord = np.stack([y, x], axis=-1)

    return new_coord


def rotation(cell_coord, nuc_coord, rna_coord, angle, ndim):
    # center point cloud
    centroid_cell = np.mean(cell_coord, axis=0)
    cell_coord_centered = cell_coord - centroid_cell
    nuc_coord_centered = nuc_coord - centroid_cell
    rna_coord_centered = rna_coord.copy().astype(np.float64)
    rna_coord_centered[:, ndim - 2:] -= centroid_cell

    # convert cartesian to polar coordinates
    cell_coord_polar = cartesian_to_polar(cell_coord_centered)
    nuc_coord_polar = cartesian_to_polar(nuc_coord_centered)
    rna_coord_polar = cartesian_to_polar(rna_coord_centered)

    # convert radians to degrees
    cell_coord_polar[:, 1] = cell_coord_polar[:, 1] * 180 / np.pi
    nuc_coord_polar[:, 1] = nuc_coord_polar[:, 1] * 180 / np.pi
    rna_coord_polar[:, ndim - 1] = rna_coord_polar[:, ndim - 1] * 180 / np.pi

    # rotate
    cell_coord_polar[:, 1] += angle
    nuc_coord_polar[:, 1] += angle
    rna_coord_polar[:, ndim - 1] += angle

    # convert degrees to radians
    cell_coord_polar[:, 1] = cell_coord_polar[:, 1] * np.pi / 180
    nuc_coord_polar[:, 1] = nuc_coord_polar[:, 1] * np.pi / 180
    rna_coord_polar[:, ndim - 1] = rna_coord_polar[:, ndim - 1] * np.pi / 180

    # convert polar to cartesian coordinates
    new_cell_coord = polar_to_cartesian(cell_coord_polar)
    new_nuc_coord = polar_to_cartesian(nuc_coord_polar)
    new_rna_coord = polar_to_cartesian(rna_coord_polar)

    # decenter point cloud
    offset_y = - new_cell_coord[:, 0].min() + stack.get_margin_value()
    offset_x = - new_cell_coord[:, 1].min() + stack.get_margin_value()
    new_cell_coord[:, 0] += offset_y
    new_cell_coord[:, 1] += offset_x
    new_nuc_coord[:, 0] += offset_y
    new_nuc_coord[:, 1] += offset_x
    new_rna_coord[:, ndim - 2] += offset_y
    new_rna_coord[:, ndim - 1] += offset_x

    return new_cell_coord, new_nuc_coord, new_rna_coord


# ### Build label ###

def build_label_clf(pattern):
    # 0 random
    # 1 foci
    # 2 intranuclear
    # 3 extranuclear
    # 4 nuclear_edge
    # 5 perinuclear
    # 6 cell_edge
    # 7 pericellular
    # (8 protrusion)

    # assign a patterns
    label = np.zeros(8, dtype=np.float32)
    if pattern == "random":
        label[0] = 1.
    if pattern == "foci":
        label[1] = 1.
    if pattern == "intranuclear":
        label[2] = 1.
    if pattern == "extranuclear":
        label[3] = 1.
    if pattern == "nuclear_edge":
        label[4] = 1.
    if pattern == "perinuclear":
        label[5] = 1.
    if pattern == "cell_edge":
        label[6] = 1.
    if pattern == "pericellular":
        label[7] = 1.

    return label


# ### TFRecords dtype ###

def _bytes_feature(value):
    """Returns a bytes_list from a string or a byte.

    Parameters
    ----------
    value : bytes or str
        Data in bytes or string.

    Returns
    -------
    feature : tf.train.Feature
        Instance of tf.train.Feature ready to be serialized in a protobuf.

    """
    # wrap bytes feature
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return feature


def _float_feature(value):
    """Returns a float_list from a float or double.

    Parameters
    ----------
    value : float32 or float64
        Data in float.
    Returns
    -------
    feature : tf.train.Feature
        Instance of tf.train.Feature ready to be serialized in a protobuf.

    """
    # wrap float feature
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    return feature


def _int64_feature(value):
    """Returns an int64_list from a boolean or an (unsigned) integer.

    Parameters
    ----------
    value : bool or int32 or uint32 or int64 or uint64
        Data in boolean or integer.

    Returns
    -------
    feature : tf.train.Feature
        Instance of tf.train.Feature ready to be serialized in a protobuf.

    """
    # wrap integer feature
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return feature


# ### Define and write examples ###

def define_example(feature):
    """Define a Example.

    Parameters
    ----------
    feature : dict
        Dictionary with the structure {"feature_name": tf.train.Feature}.

    Returns
    -------
    example : tf.train.Example
        Protobuf to serialized.

    """
    # define example
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example


def define_sequence_example(feature_list, feature_context):
    """Define a SequenceExample.

    Parameters
    ----------
    feature_list : dict
        Dictionary with the structure {"feature_name": tf.train.FeatureList}.
    feature_context : dict
        Dictionary with the structure {"feature_name": tf.train.Feature}.

    Returns
    -------
    example : tf.train.SequenceExample
        Protobuf to serialized.

    """
    # define example
    example = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(feature_list=feature_list),
        context=tf.train.Features(feature=feature_context))
    return example


def write_tfrecord(examples, path):
    if not isinstance(examples, list):
        examples = [examples]

    # open file to write tfrecords inside
    with tf.io.TFRecordWriter(path) as writer:
        for example in examples:
            serialized_example = example.SerializeToString()
            writer.write(serialized_example)

    return


# ### Example classification ###

def build_example_clf(
        rna_coord,
        cell_coord,
        nuc_coord,
        cell_mask,
        nuc_mask,
        pattern=None,
        add_cluster=False,
        voxel_size=(100, 100, 100),
        add_morphology=False,
        n_coord_cell=300,
        n_coord_nuc=100,
        add_distance=False,
        normalized=True,
        random_rotation=False,
        name=None):
    # compute features
    (input_coordinate,
     input_cluster,
     input_morphology,
     input_distance) = build_features(
        rna_coord,
        cell_coord,
        nuc_coord,
        cell_mask,
        nuc_mask,
        add_cluster=add_cluster,
        voxel_size=voxel_size,
        add_morphology=add_morphology,
        n_coord_cell=n_coord_cell,
        n_coord_nuc=n_coord_nuc,
        add_distance=add_distance,
        normalized=normalized,
        random_rotation=random_rotation)

    # format input features
    n = len(input_coordinate)
    if input_cluster is None:
        input_cluster = np.zeros((n, 1), dtype=np.float32)
    if input_morphology is None:
        input_morphology = np.zeros((n, 2), dtype=np.float32)
    if input_distance is None:
        input_distance = np.zeros((n, 2), dtype=np.float32)

    # prepare label
    label = build_label_clf(pattern)

    # define example
    example = define_example_clf(
        input_coordinate=input_coordinate,
        input_cluster=input_cluster,
        input_morphology=input_morphology,
        input_distance=input_distance,
        label=label,
        name_sample=name)

    return example


def define_example_clf(
        input_coordinate,
        input_cluster,
        input_morphology,
        input_distance,
        label,
        name_sample=None):
    # prepare data
    if name_sample is not None:
        name_sample = name_sample.encode("utf-8")
    length_input = len(input_coordinate)
    raw_input_coordinate = input_coordinate.ravel().tobytes()
    raw_input_cluster = input_cluster.ravel().tobytes()
    raw_input_morphology = input_morphology.ravel().tobytes()
    raw_input_distance = input_distance.ravel().tobytes()
    raw_label = None
    if label is not None:
        raw_label = label.ravel().tobytes()

    # build dictionary with tf.train.Feature
    feature = {
        "raw_input_coordinate": _bytes_feature(raw_input_coordinate),
        "raw_input_cluster": _bytes_feature(raw_input_cluster),
        "raw_input_morphology": _bytes_feature(raw_input_morphology),
        "raw_input_distance": _bytes_feature(raw_input_distance),
        "length_input": _int64_feature(length_input)}
    if label is not None:
        feature["raw_label"] = _bytes_feature(raw_label)
    if name_sample is not None:
        feature["name_sample"] = _bytes_feature(name_sample)

    # build example to serialized
    example = define_example(feature)

    return example


# ### Parse examples ###

def _parse_example_proto(example_proto, feature_description):
    # parse the input tf.train.Example proto
    return tf.io.parse_single_example(example_proto, feature_description)


def parse_example_clf(
        example,
        add_label=False,
        add_metadata=False,
        input_dimension=3):
    # parse example
    example_description = {
        "raw_input_coordinate": tf.io.FixedLenFeature([], tf.string),
        "raw_input_cluster": tf.io.FixedLenFeature([], tf.string),
        "raw_input_morphology": tf.io.FixedLenFeature([], tf.string),
        "raw_input_distance": tf.io.FixedLenFeature([], tf.string),
        'length_input': tf.io.FixedLenFeature([], tf.int64)}
    if add_label:
        example_description["raw_label"] = tf.io.FixedLenFeature([], tf.string)
    if add_metadata:
        example_description["name_sample"] = tf.io.VarLenFeature(tf.string)
    parsed_example = _parse_example_proto(example, example_description)

    # format features
    name_sample = None
    if add_metadata:
        name_sample = parsed_example['name_sample']
        name_sample = tf.sparse.to_dense(name_sample)
    label = None
    if add_label:
        label = parsed_example['raw_label']
        label = tf.io.decode_raw(label, out_type=float)
    length_input = parsed_example['length_input']
    shape_coordinate = tf.stack([length_input, 3])
    input_coordinate = parsed_example['raw_input_coordinate']
    input_coordinate = tf.io.decode_raw(input_coordinate, out_type=float)
    input_coordinate = tf.reshape(input_coordinate, shape_coordinate)
    if input_dimension == 2:
        input_coordinate = input_coordinate[:, 1:]
    shape_cluster = tf.stack([length_input, 1])
    input_cluster = parsed_example['raw_input_cluster']
    input_cluster = tf.io.decode_raw(input_cluster, out_type=float)
    input_cluster = tf.reshape(input_cluster, shape_cluster)
    shape_morphology = tf.stack([length_input, 2])
    input_morphology = parsed_example['raw_input_morphology']
    input_morphology = tf.io.decode_raw(input_morphology, out_type=float)
    input_morphology = tf.reshape(input_morphology, shape_morphology)
    shape_distance = tf.stack([length_input, 2])
    input_distance = parsed_example['raw_input_distance']
    input_distance = tf.io.decode_raw(input_distance, out_type=float)
    input_distance = tf.reshape(input_distance, shape_distance)

    # format output
    features = {"input_coordinate": input_coordinate,
                "input_cluster": input_cluster,
                "input_morphology": input_morphology,
                "input_distance": input_distance}
    if add_metadata:
        features["name_sample"] = name_sample
    if add_label:
        target = {
            "label": label}
        return features, target

    return features


# ### Build datasets ###

def build_dataset_from_tfrecords(path, parser, parallel_calls):
    # read and parse tfrecords
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parser, num_parallel_calls=parallel_calls)

    return dataset


def build_dataset(
        path_tfrecords_directories,
        function_parse,
        batch_size,
        buffer=2048,
        n_jobs=tf.data.AUTOTUNE,
        low_length=50,
        high_length=950):
    # get paths tfrecords
    path_tfrecords = []
    if not isinstance(path_tfrecords_directories, list):
        path_tfrecords_directories = [path_tfrecords_directories]
    for path_folder in path_tfrecords_directories:
        for filename in os.listdir(path_folder):
            if ".tfrecords" not in filename:
                continue
            path_tfrecord = os.path.join(path_folder, filename)
            path_tfrecords.append(path_tfrecord)
    np.random.shuffle(path_tfrecords)

    # build dataset
    dataset = build_dataset_from_tfrecords(
        path_tfrecords, function_parse,
        parallel_calls=n_jobs)

    # shuffle dataset
    dataset = dataset.shuffle(buffer)

    # bucket and batch dataset
    if batch_size == 1:
        dataset = dataset.batch(batch_size)
    else:
        def _element_length_func(x, y):
            return tf.shape(x["input_coordinate"])[0]
        boundaries = [i + 1 for i in range(low_length, high_length, 50)]
        batches = [batch_size] * (len(boundaries) + 1)
        dataset = dataset.bucket_by_sequence_length(
            element_length_func=_element_length_func,
            bucket_boundaries=boundaries,
            bucket_batch_sizes=batches,
            no_padding=True)

    # prefetch dataset
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
