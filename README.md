<img align="left" width="100" height="100" src="minML_icon.png">

# miniML: A deep learning framework for synaptic event detection

[![minimal Python version](https://img.shields.io/badge/Python-3.9_3.10-grey.svg?style=for-the-badge&logo=python&labelColor=3670A0&logoColor=white)](https://www.anaconda.com/download/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![DOI](https://img.shields.io/badge/DOI-10.1101/2023.11.02.565316-grey.svg?style=for-the-badge&logo=doi&labelColor=green&logoColor=white)](https://www.biorxiv.org/content/10.1101/2023.11.02.565316)


This repository contains the code described in the following publication:  
O'Neill P.S., Baccino-Calace M., Rupprecht P., Friedrich R.W., Müller, M., and Delvendahl, I. 
(2024) Deep learning-based synaptic event detection. _eLife_ ([doi:10.7554/eLife.98485.1](https://doi.org/10.7554/eLife.98485.1))  


------------
### Update 03.07.2024:


Part of the downstream processing as well as parts of the plot functionality was changed. If you have been using miniML things should work as per usual, with the following exceptions:
- Plot functionalities now are in a separate class and need to be generated slightly different. Please refer to the section "Inspect the results" in the tutorial.ipynb file for details.
- The quantitative results for amplitude etc. may differ slightly from the previous version. Reason is the way we process the data as well as the filtering (see below). Previously, event location, event peak etc. were determined in resampled data (if win_size was not 600) and then mapped to the raw data. Now we work with the raw data in this step. Therefore, the exact values may differ.
- The convolve_win parameter may need to be adjusted. Filtering was previously done in resampled data, so you may need to adjust the filter settings. You can visually check for good settings using the plot_gradient_search() method. See also section "Inspect the results" in the tutorial.ipynb file for details.
- The save_to_csv() method has been updated to be consistent with the other file formats. It previously took a path as argument, now it takes a filename.
------------

### 🧠 ABOUT

miniML is a deep-learning-based tool to detect synaptic events in 1d timeseries data. It uses a CNN-LSTM network architecture that was trained using a large dataset of synaptic events from cerebellar mossy fiber to granule cell synapses. 

In this repository, we provide pretrained models and Python code to run model inference on recorded data. In addition, an application example (cerebellar granule cell recording) is included.


### 💻 INSTALLATION

To use miniML, clone the GitHub Repositiory and install the requirements. The Python dependencies are: sklearn, matplotlib, h5py, pandas, numpy, scipy, tensorflow, pyabf. To install all dependencies using pip, run the following command in your Python environment:

`pip install -r requirements.txt`

miniML can be run on a GPU to speed model inference. Either CUDA or tensorflow-metal are required for GPU use. Installation instructions for these requirements may depend on the specific hardware and OS and can be found online.

**Update June 2024:** The release of TensorFlow 2.16 and Keras 3 introduced breaking changes that raise an error when loading models trained with earlier TensorFlow versions. To avoid this, it is recommended to use TensorFlow 2.14 or 2.15.


### ⏱ RUNNING MINIML

First, a miniML *MiniTrace* object needs to be created containing 1d timeseries data. Currently, miniML supports direct loading from HEKA .dat files, Axon .abf files as well as HDF .h5 files. The *Trace* object features discrete methods for loading from these file formats (e.g., **MiniTrace.from_h5_file()**). Data in other file formats need to be imported as Python objects.

Next, a miniML *EventDetection* object is initiated. Here, one needs to specify a miniML model file to use as well as the *Trace* object to operate on. 

Finally, model inference can be run using the **detect_events()** method. This method will run miniML over the given data using the specified model. Runtime will depend on data length.

Detected events are analyzed and descriptive statistics can subsequently be saved in different formats (.pickle, .h5, .csv).


### 💡 EXAMPLE

The folder "example_data/" contains an example recording from a cerebellar mossy fiber to granule cell synapse together with a commented Jupyter Notebook ([tutorial](tutorial.ipynb)) illustrating the use of miniML.


### 📢 UPDATE INFORMATION

With the latest update, we include all the files required to generate and score the data to train a new model and use the model on your data. For the actual training, please refer to the following links to Kaggle:  

[1 - Transfer learning](https://www.kaggle.com/code/philipponeill/miniml-transfer-learning)  
[2 - Full training](https://www.kaggle.com/code/philipponeill/miniml-full-training)  
[3 - Training dataset](https://www.kaggle.com/datasets/philipponeill/miniml-training-data)  

We had to make one change that will impact scripts written for the previous miniML version. We split the *direction* parameter (used by the miniML **detect_events()** method) into two separate parameters: *event_direction* and *training_direction* (refer to [miniML documentation](miniML_documentation.pdf) for details). In practice, this means that you may have to rename the parameter *direction* to *event_direction* in existing scripts.

Please feel free to contact us in case of questions, either via email, or by opening an issue here on GitHub (chances are that other people have the same question).


### ✉️ CONTACT
philippsean.oneill@uzh.ch or igor.delvendahl@uzh.ch