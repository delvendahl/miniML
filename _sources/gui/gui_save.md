# Save analysis results

After running miniML analysis, the results can be saved to a file.
miniML currently supports saving the results to text (.csv), Pickle (.pickle) and HDF5 (.h5) files.

## Saving to file

Clicking the **Save** button opens a choose file dialog. The default name is the name of the recording. 
The file can be saved as a .csv, .pickle or .h5 file.

### CSV

When saving as CSV, the average results are saved in the *_avg.csv file and the individual event statistics in the *_individual.csv file. Individual event statistics include event location (in samples), model prediction value ("score"), amplitude, charge, risetime and decay time. The average values saved are mean event amplitude (in data units), standard deviation of amplitudes (in data units), median event amplitude (in data units), mean charge (in data units x s), mean 10-90 risetime (in s), mean decay time (in s), mean halfwidth (in s), decay time constant from an exponential fit to the average event waveform (in s), and event frequency (in Hz).

### HDF5

When saving as HDF5, the results are saved in a single HDF5 file. The file contains the following datasets:
- events: all detected events
- event_params: event_locations, event_scores, event_amplitudes, event_charges, event_risetimes, event_halfdecays, event_bsls
- event_statistics: amplitude_average, amplitude_stdev, amplitude_median, charge_mean, charge_median, risetime_mean, risetime_median, decaytime_mean, decaytime_median, halfwidth_mean, halfwidth_median, decay_from_fit, frequency
The file contains the following attributes: amplitude_unit, recording_time, source_filename, miniml_model, miniml_model_threshold, stride, window, event_direction, filter_factor, gradient_convolve_win, relative_prominence, deleted_events

### PICKLE

When saving as Pickle, the results are saved in a single Pickle file. The file contains the following dictionary:
- event_location_parameters
- individual_values
- average_values
- average_event_properties
- average_event_fit
- metadata
- events

```{seealso}
For more information, see the miniml.py save_to_pickle() function.
```
