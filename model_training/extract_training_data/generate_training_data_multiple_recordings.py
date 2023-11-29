'''
Script to automatically extract training data using template matching.

Idea:
Extract events using a relatively high threshold. Extract noise by extracting
whatever data is not picked up by a very low threshold template matching.
Extract a mix of small-amplitude events and potential FPs using an intermediate threshold.


What can go wrong and what can you do?
- If the template does not capture the actual events, events might be missed.
    If that is the case, you can use the jupyter "generate_training_data_single_recording.ipynb" to
    check the template and adjust it to your needs.

- similarly, if you have heterogeneous event kinetics in one recording, a single template
  might not find all of them.
    In this case, you can always adjust the mid threshold. There will a lot more False positives,
    and it will take slightly longer to score later, but you should be able to automatically
    extract most events in this manner.

- High event frequency can lead to very limited data being available for the pure noise

- One cell completely dominates the training data due to above average frequency;
  if e.g. you have on average 30 events, but one cell has 300, and you only look at four
  total, the one cell will contribute 300 events / 390. In this case, you can extract
  the data and only take a subset of the events and noise from the dominating cell afterwards.
'''

import sys
sys.path.append('../../core/')

from miniML import MiniTrace
import template_matching as tm
import numpy as np
from scipy import signal
import h5py
import copy
import os
np.random.seed(0)

path = './example_recordings/'

cell_label, idx, events, scores = [], [], [], []

# Set the different thresholds.
threshold_high = -4.5
threshold_mid = -3
threshold_low = -1.6

for myfile in os.listdir(path):
    if myfile.endswith('.h5'):
        print(f'extracting from: {myfile}')
        filename = path+myfile
        scaling = 1e12
        unit = 'pA'

        # get from h5 file
        trace = MiniTrace.from_h5_file(filename=filename,
                                    tracename='mini_data',
                                    scaling=scaling,
                                    unit=unit)


        # For template matching it can be useful to filter the data. We use a Hann window here.
        filter_data = True
        if filter_data:
            win = signal.windows.hann(20)
            tmplt_trace = signal.convolve(trace.data, win, mode='same') / sum(win)
        else:
            tmplt_trace = trace.data

        win_size = 600

        # roughly estimate possible event shape based on window size
        baseline = (win_size/8) * trace.sampling
        duration = int(win_size*1/3) * trace.sampling
        t_rise = (baseline+(win_size * trace.sampling))/40
        t_decay = (baseline+(win_size * trace.sampling))/15

        template = tm.make_template(t_rise=t_rise, t_decay=t_decay, baseline=baseline, duration=duration, sampling=trace.sampling)


        # Run template matching with a high threshold to extract events with high confidence.
        matching = tm.template_matching(tmplt_trace, template, threshold=threshold_high)
        print(f'found {len(matching.indices)} events with high threshold')
        
        event_counter = 0
        for ind, location in enumerate(matching.indices):
            if location < trace.data.shape[0] - int(win_size*1.5):
                event = copy.deepcopy(trace.data[location:location+win_size])
                event -= np.mean(event[:int(win_size/10)])
                
                cell_label.append(filename)
                idx.append(location)
                events.append(event)
                scores.append(1)
                event_counter += 1
        
        print(f'{event_counter} events extracted')


        # generate list with all indices of +/- (win_size/30) points of the previously found events to prevent duplicates
        idx_range = []
        buffer = int(win_size/30)
        for my_ind in idx:
            idx_range += list(range(my_ind-buffer, my_ind+buffer))

        # Run tmplt matching with a relatively low threshold to extract FPs and small events.
        matching = tm.template_matching(tmplt_trace, template, threshold=threshold_mid)
        print(f'found {len(matching.indices)} events with mid threshold')
        unclear_counter = 0
        for ind, location in enumerate(matching.indices):
            if location < trace.data.shape[0] - int(win_size*1.5):
                if location not in idx_range:
                    event = copy.deepcopy(trace.data[location:location+win_size])
                    event -= np.mean(event[:int(win_size/10)])
                    
                    cell_label.append(filename)
                    idx.append(location)
                    events.append(event)
                    scores.append(2)
                    unclear_counter += 1

        print(f'{unclear_counter} unclear events extracted')


        # Run tmplt matching with a very low threshold. Remaining parts of the trace should be event free.
        matching = tm.template_matching(tmplt_trace, template, threshold=threshold_low)
        print(f'found {len(matching.indices)} events with low threshold')

        event_free_indices = []
        for i in range(matching.indices.shape[0]-1):
            start = matching.indices[i] + win_size
            end = matching.indices[i+1] - win_size
            if end - start > 0:
                event_free_indices.append(np.arange(start, end))

        event_free_indices = np.concatenate(event_free_indices)
        unique_stretches = []
        for ind, i in enumerate(event_free_indices):
            if ind==0:
                unique_stretches.append(i)
                next_possible = i+win_size
            
            if i < next_possible:
                pass
            else:
                unique_stretches.append(i)
                next_possible = i+win_size

        # Extract unique stretches to prevent overlap and redundancy in the data
        if len(unique_stretches) <= event_counter:
            inds = np.array(unique_stretches)
        else:
            inds = np.random.choice(np.array(unique_stretches), event_counter, replace=False)

        # Extract events
        noise_counter = 0
        for location in sorted(inds):
            event = copy.deepcopy(trace.data[location:location+win_size])
            event -= np.mean(event[:int(win_size/10)])
            cell_label.append(filename)
            idx.append(location)
            events.append(event)
            scores.append(0)
            noise_counter += 1

        print(f'{noise_counter} noise stretches extracted')
        print(f'\n\n')



x = np.array(events)
y = np.array(scores)
indices = np.array(idx)

# Save the data. For training, only events and scores are required, but it can be helpful to know from which trace
# and where exactly in the trace the event came from (e.g. in case one cell dominates the dataset and you want to
# exclude some of the data from that specific cell to keep the dataset balanced).
save_dataset = './output/multiple_files_example_training_data.h5'
if save_dataset:
    with h5py.File(save_dataset, 'w') as f:
        f.create_dataset("events", data=x)
        f.create_dataset("scores", data=y)
        f.create_dataset("raw_indices", data=indices)
        f.create_dataset("cell_label", data=np.array(cell_label, dtype='S'))