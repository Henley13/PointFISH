# PointFISH

[![License](https://img.shields.io/badge/license-BSD%203--Clause-green)](https://github.com/Henley13/PointFISH/blob/main/LICENSE)

This repository gathers the code used for the following paper published in ECCV 2022 BioImage Computing workshop:

__Title:__ PointFISH: learning point cloud representations for RNA localization patterns

__Authors:__ [Arthur Imbert](mailto:Arthur.Imbert@minesparis.psl.eu)<sup>1,2,3\*</sup>, Florian Mueller<sup>4, 5</sup>, [Thomas Walter](mailto:Thomas.Walter@minesparis.psl.eu)<sup>1,2,3\*</sup>

><sup>1</sup>Centre for Computational Biology, Mines Paris, PSL University, Paris, France  
<sup>2</sup>Institut Curie, PSL University, Paris, France  
<sup>3</sup>INSERM, U900, Paris, France  
<sup>4</sup>Imaging and Modeling Unit, Institut Pasteur and UMR 3691 CNRS, Paris, France  
<sup>5</sup>C3BI, USR 3756 IP CNRS, Paris, France  
>
><sup>\*</sup>To whom correspondence should be addressed.

## Pipeline

Below, we list the different steps performed in the paper:

1. simulate_patterns.py
2. build_tfrecords_clf.py
3. train_clf.py