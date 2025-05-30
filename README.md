<img align="left" width="100" height="100" src="minML_icon.png">

# miniML: A deep learning framework for synaptic event detection

[![minimal Python version](https://img.shields.io/badge/Python-3.9_3.10-grey.svg?style=for-the-badge&logo=python&labelColor=3670A0&logoColor=white)](https://www.anaconda.com/download/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![DOI](https://img.shields.io/badge/DOI-10.7554/eLife.98485-grey.svg?style=for-the-badge&logo=doi&labelColor=green&logoColor=white)](https://doi.org/10.7554/eLife.98485)


This repository contains the code described in the following publication:  
O'Neill P.S., Baccino-Calace M., Rupprecht P., Lee, S., Hao, Y.A., Lin, M.Z., Friedrich R.W., Müller, M., and Delvendahl, I. 
(2025) A deep learning framework for automated and generalized synaptic event analysis. _eLife_ 13:RP98485 ([doi:10.7554/eLife.98485](https://doi.org/10.7554/eLife.98485))  


### 🧠 ABOUT

miniML is a deep-learning-based tool to detect synaptic events in 1d time-series data. It uses a CNN-LSTM network architecture that was trained using a large dataset of synaptic events (miniature excitatory postsynaptic currents) from cerebellar mossy fiber to granule cell synapses. 

In this repository, we provide documentation, pre-trained models, and Python code to run model inference on recorded data. In addition, an application example (cerebellar granule cell mEPSC recording) is included.


### 📢 UPDATES

#### 20 March 2025

The latest update introduces updated GUI functionality. The GUI now includes a new version of the EventViewer that allows for more detailed inspection of detected events. The EventViewer now includes infomration on each event, the location in the datatrace, as well as the ability to delete individual events. We also updated the filter and cutout functionality, with enhanced user control.

#### 8 January 2025
We have updated the documentation for miniML. The latest version of the documentation is available [here](https://delvendahl.github.io/miniML/intro.html).


### 💻 INSTALLATION

To use miniML, clone the GitHub Repository and install the requirements. miniML has been tested with Python 3.9 and 3.10. The Python dependencies are: sklearn, matplotlib, h5py, pandas, numpy, scipy, tensorflow, pyabf. To install all dependencies using pip, run the following command in your Python environment:

`pip install -r requirements.txt`

>[!IMPORTANT]
>The release of TensorFlow 2.16 and Keras 3 introduced breaking changes that raise an error when loading models trained with earlier TensorFlow versions. To avoid this, it is recommended to use TensorFlow 2.14 or 2.15.

> [!NOTE]
>miniML can be run on a GPU to speed model inference. Either CUDA or tensorflow-metal are required for GPU use. Installation instructions for these requirements may depend on the specific hardware and OS and can be found online.


To use miniML with a graphical user interface (GUI), you need to install the additional dependencies from the requirements_gui.txt file.  

`pip install -r requirements_gui.txt`


### ⏱ RUNNING MINIML

#### Analysis workflow in Python
First, a miniML *MiniTrace* object needs to be created containing 1d timeseries data. Currently, miniML supports direct loading from HEKA .dat files, Axon .abf files as well as HDF .h5 files. The *Trace* object features discrete methods for loading from these file formats (e.g., **MiniTrace.from_h5_file()**). Data in other file formats need to be imported as Python objects.

Next, a miniML *EventDetection* object is initiated. Here, one needs to specify a miniML model file to use as well as the *Trace* object to operate on. 

miniML model inference can then be run using the **detect_events()** method. This method will run miniML over the given data using the specified model. Runtime will depend on data length. 

Following event detection, the individual detected events are analyzed and descriptive statistics are calculated for the recording.

miniML includes several plotting methods. They can be found in the **miniML_plots** class in `miniML_plot_functions.py`. A detection object has to be passed as data argument. 

Event data and statistics can be saved in different formats (.pickle, .h5, .csv).

> [!TIP]
>miniML can also be used via a GUI. To use the GUI, execute the miniml_gui.py file (located in the "core/" folder). The GUI allows easy loading of data, pre-processing (filtering, detrending etc.) and model inference. Found events can be inspected and deleted, if desired. The GUI can also be used to save results to a PICKLE, CSV or HDF5 file.


### 💡 EXAMPLE

The folder "example_data/" contains an example recording from a cerebellar mossy fiber to granule cell synapse. To use miniML on this data, run the commented example Jupyter Notebook ([tutorial](docs/general/tutorial.ipynb)) illustrating the use of miniML.


### 📚 DOCUMENTATION

Detailed documentation for miniML can be found [here](https://delvendahl.github.io/miniML/intro.html).


### 📦 MODELS

The documentation includes notebooks showing how to train miniML models. You can also use the following links to Kaggle to train mminiML models on the cloud:  

[1 - Transfer learning](https://www.kaggle.com/code/philipponeill/miniml-transfer-learning)  
[2 - Full training](https://www.kaggle.com/code/philipponeill/miniml-full-training)  
[3 - Training dataset](https://www.kaggle.com/datasets/philipponeill/miniml-training-data)  

The repository contains trained models for several event detection scenarios, as outlined in the associated paper. If you have trained a model that could be useful for other researchers, please consider opening a pull request or get in touch with us in order to add it to the repository.


### 📝 CITATION

If you use miniML in your work, please cite:
```BibTeX
@article{ONeill2025,
  article_type = {journal},
  title = {A deep learning framework for automated and generalized synaptic event analysis},
  author = {O'Neill, Philipp S and Baccino-Calace, Martín and Rupprecht, Peter and Lee, Sungmoo and Hao, Yukun A and Lin, Michael Z and Friedrich, Rainer W and Mueller, Martin and Delvendahl, Igor},
  volume = 13,
  year = 2025,
  month = {mar},
  pub_date = {2025-03-05},
  pages = {RP98485},
  citation = {eLife 2025;13:RP98485},
  doi = {10.7554/eLife.98485},
  url = {https://doi.org/10.7554/eLife.98485},
}
```

### 🙏 ACKNOWLEDGEMENTS

The development of miniML was funded by the Swiss National Science Foundation, the University of Zurich Research Talent Development Fund, and the German Research Foundation.

The GUI uses Material Icons by Google (https://fonts.google.com/icons).

The Python code to read HEKA Patchmaster files was adapted from https://github.com/campagnola/heka_reader

The template matching implementation was adapted from https://github.com/samuroi/SamuROI

### 🐛 ISSUES

Please feel free to contact us in case of questions, either via email, or by opening an [Issue](https://github.com/delvendahl/miniML/issues) here on GitHub.


### ✉️ CONTACT
philipp.oneill@physiologie.uni-freiburg.de or igor.delvendahl@physiologie.uni-freiburg.de
