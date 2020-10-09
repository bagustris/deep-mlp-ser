# Repository for my paper: *Deep Multilayer Perceptrons for Dimensional Speech Emotion Recognition*

by
Bagus Tris Atmaja,    
Masato Akagi  
Email: bagus@ep.it.ac.id

> This is a repository for above paper consist of 
> Python codes, Latex codes, and figures.

This paper has been accepted for publication in *APSIPA ASC 2020*.

![](fig/deep_mlp.png)

*Architecture of proposed method.*


## Abstract

> Modern deep learning architectures are ordinarily
performed on high-performance computing facilities due to the
large size of the input features and complexity of its model.
This paper proposes traditional multilayer perceptrons (MLP)
with deep layers and small input size to tackle that computation
requirement limitation. The result shows that our proposed deep
MLP outperformed modern deep learning architectures, i.e.,
LSTM and CNN, on the same number of layers and value of
parameters. The deep MLP exhibited the highest performance
on both speaker-dependent and speaker-independent scenarios
on IEMOCAP and MSP-IMPROV corpus.


## Software implementation


All source code used to generate the results and figures in the paper are in
the `code` folder.
The calculations and figure generation are by 
running each Python code.
The data used in this study is provided in `data` and the sources for the
manuscript text and figures are in `latex`.
Results generated by the code are saved in `results`, if any.
See the `README.md` files in each directory for a full description.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/bagustris/deep_mlp_ser.git

or [download a zip archive](https://github.com/bagustris/deep_mlp_ser).



## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.

We use `pip` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    pip3.6 venv REPO_NAME


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate REPO_NAME

To reproduce result in , run each related Python file.  


## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
EUSIPCO 2020.


## Citation
```
Atmaja, Bagus Tris and Akagi, Masato. Deep Multilayer Perceptrons for Dimensional
Speech Emotion Recognition, 2020.
```

