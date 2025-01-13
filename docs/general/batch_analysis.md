# Batch analysis

miniML can be easily run on multiple traces/recordings in a batch. This feature is useful for large datasets, where running miniML on a single trace individually can take a long time.

The simplest approach is to run miniML in a loop over all recordings and save the results of each analysis to a file. 

```{tip}
For batch analysis, we recommend to create  a model instance outside of the loop and pass it to the miniML detection object. This will save time and memory.
```

In the following, we provide example code for the analysis of multiple recordings. You can set the verbose parameter to 0 to prevent output from being printed (recommended for large datasets).

```python
import tensorflow as tf
from pathlib import Path

file_list = ['recording_1.dat', 'recording_2.dat', 'recording_3.dat']
raw_data_folder = 'data'
results_folder = 'results'

scaling = 1e12
unit = 'pA'
pgf_name = 'conti VC'
miniml_model = tf.keras.models.load_model('GC_lstm_model.h5')
window_size = 600

for filename in file_list:
    filepath = Path(folder_name) / filename
    # load data from file
    trace = MiniTrace.from_heka_file(filename=filepath,
                                     rectype=pgf_name,
                                     exclude_series=None,
                                     scaling=scaling,
                                     unit=unit)

    # create miniML detection object
    detection = EventDetection(data=trace,
                               model=miniml_model
                               model_threshold=0.5,
                               window_size=window_size,
                               batch_size=512,
                               event_direction='negative',
                               verbose=0)

    # run analysis
    detection.detect_events(eval=True,
                            peak_w=5,
                            rel_prom_cutoff=0.25,
                            convolve_win=20,
                            gradient_convolve_win=40)
    
    # Save results to file
    detection.save_to_pickle(filename=f'results_folder/{filepath.stem}.pickle', 
                             include_prediction=False, 
                             include_data=False)
```

Of course, you can also collect the results of the analysis in a custom object and use them for further analysis and/or saving to file. In the following example, we collect the results of the analysis in a pandas dataframe.

```{hint}
The miniML EventStats class (found in miniML.py) contains all the results of the analysis and includes methods to caculate, e.g., mean and median values.
```

```python
import pandas as pd

# code from the above example goes here
#
#

# create empty dataframe
my_df = pd.DataFrame()

for i, filename in enumerate(file_list):

    # code as in the above example
    #
    #

    my_df.loc[i, 'recording'] = filename
    my_df.loc[i, 'recording_time'] = trace.data.shape[0] * trace.sampling,
    my_df.loc[i, 'amplitude_mean'] = detection.event_stats.mean(detection.event_stats.amplitudes)
    my_df.loc[i, 'amplitude_median'] = detection.event_stats.median(detection.event_stats.amplitudes)
    my_df.loc[i, 'charge_mean'] = detection.event_stats.mean(detection.event_stats.charges)
    my_df.loc[i, 'decay_mean'] = detection.event_stats.mean(detection.event_stats.halfdecays)
    my_df.loc[i, 'risetime_mean'] = detection.event_stats.mean(detection.event_stats.risetimes)
    my_df.loc[i, 'frequency'] = detection.event_stats.frequency()

my_df.to_csv('results.csv')

```


```{important}
When analysing many recordings, do not use the plotting functions in a loop, in particular if you are runnning the analysis in a jupyter notebook. Matplotlib will accumulate data in memory, causing severe slowdowns. Smaller batches should be used instead if plots are required.
```
