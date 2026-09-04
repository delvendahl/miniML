# Installation

miniML is written in Python and is distrubuted as a Python package. It can be installed from PyPI using pip (recommendend for most users), or locally by cloning the GitHub repository and installing it with, e.g., pip.

```bash
pip install miniml-detect[gui]
```

```{hint} 
We recommend creating a virtual environment for miniML using Python version 3.11.
```

The Python dependencies for miniML are: 
- sklearn
- matplotlib
- h5py
- numpy
- scipy
- tensorflow
- pyabf
- pyheka
- ruptures
- pyQT5 (gui only)
- qt-material (gui only)
- pyqtgraph (gui only)


To install miniML locally using pip, clone the project's GitHub Repository, cd to the miniML folder, and run the following command in your Python environment:

```bash
git clone https://github.com/delvendahl/miniML.git
cd miniML
pip install ".[gui]"
```

You can also install miniML in editable mode, which allows you to make changes to the code and have them reflected immediately without needing to reinstall the package. To do this, run:

```bash
pip install -e ".[gui]"
```



This will install everything you need to run miniML as a python package.

To install miniML without the GUI option, run:

```bash
pip install miniml-detect
```

or locally:

```bash
pip install .
```

```{important}
The release of TensorFlow 2.16 and Keras 3 introduced breaking changes that raise an error when loading models trained with earlier TensorFlow versions. To avoid this, it is recommended to use TensorFlow 2.14 or 2.15.
```

miniML can be run on a GPU to speed model inference. 
Either CUDA or tensorflow-metal are required for GPU use. Installation instructions for these requirements
may depend on the specific hardware and OS and can be found online.

