# Installation

miniML is written in Python. To use miniML, clone the project's GitHub Repository and install the requirements. 

`git clone https://github.com/delvendahl/miniML.git`

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


To install miniML using pip, cd to the miniML folder and run the following command in your Python environment:
```bash
pip install .
```

This will install everything you need to run miniML as a python package.

To install miniML with the GUI option, run:
```bash
pip install ".[gui]"
```

```{important}
The release of TensorFlow 2.16 and Keras 3 introduced breaking changes that raise an error when loading models trained with earlier TensorFlow versions. To avoid this, it is recommended to use TensorFlow 2.14 or 2.15.
```

miniML can be run on a GPU to speed model inference. 
Either CUDA or tensorflow-metal are required for GPU use. Installation instructions for these requirements
may depend on the specific hardware and OS and can be found online.

