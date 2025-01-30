from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
from miniML import EventDetection, mEPSC_template
from scipy.ndimage import maximum_filter1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class miniML_plots():
    '''miniML plotting class. Allows calling multiple standard plots based on miniML EventDetection objection
    Parameters
    ----------
    data: miniML EventDetection object
        Event detection and associated values and parameters to be plotted.  
    '''
    def __init__(self, data: EventDetection) -> None:
        self.detection = data
        self.main_trace_color = '#014182' # Blue variant to be used in plots.
        self.orange_color = '#f0833a' # orange variant to be used in plots.
        self.red_color = '#a90308'
        self.green_color = '#287c37'

    def plot_gradient_search(self):
        fig, axs = plt.subplots(3, sharex=True, num='gradient search')

        mini_trace = self.detection.trace.data
        filtered_trace = self.detection.hann_filter(self.detection.trace.data, filter_size=self.detection.convolve_win)

        filtered_prediction = maximum_filter1d(self.detection.prediction, size=int(5*self.detection.interpol_factor), origin=-2)

        axs[0].plot(filtered_prediction, self.main_trace_color)
        axs[0].scatter(self.detection.start_pnts, self.detection.prediction[self.detection.start_pnts], c=self.red_color, zorder=2, label='start points')
        axs[0].scatter(self.detection.end_pnts, self.detection.prediction[self.detection.end_pnts], c=self.green_color, zorder=2, label='end points')
        axs[0].legend(loc='upper right')

        axs[1].plot(mini_trace, c='k', alpha=0.4)
        axs[1].plot(filtered_trace, c=self.main_trace_color)

        axs[1].scatter(self.detection.event_locations, filtered_trace[self.detection.event_locations], c=self.orange_color, zorder=2, label='event locations')
        axs[1].scatter(self.detection.start_pnts, filtered_trace[self.detection.start_pnts], c=self.red_color, zorder=2)
        axs[1].scatter(self.detection.end_pnts, filtered_trace[self.detection.end_pnts], c=self.green_color, zorder=2)
        axs[1].legend(loc='upper right')

        axs[2].plot(self.detection.gradient, c='k', alpha=0.4, label='gradient')
        axs[2].plot(self.detection.smth_gradient, self.main_trace_color, label='filtered gradient')
        axs[2].axhline(self.detection.grad_threshold, c=self.orange_color, ls='--', label='threshold (4x std of noise)')
        axs[2].legend(loc='upper right')
        plt.show()

    def plot_event_overlay(self) -> None:
        '''
        plot the average event waveform overlayed on top of the individual events
        plus the fitted event.
        '''
        if not self.detection.events_present():
            return
        
        fig = plt.figure('Event average and fit')
        plt.plot(np.arange(0, self.detection.events.shape[1]) * self.detection.trace.sampling, self.detection.events.T, c=self.main_trace_color, alpha=0.3)
        
        # average
        ev_average = np.mean(self.detection.events, axis=0)
        plt.plot(np.arange(0, self.detection.events.shape[1]) * self.detection.trace.sampling, ev_average, c=self.red_color, linewidth='3', label='average event')
        
        # fit
        fitted_ev = mEPSC_template(np.arange(0, self.detection.events.shape[1]-int(self.detection.window_size/6)) * self.detection.trace.sampling, 
                                   *self.detection.fitted_avg_event.values())

        plt.plot(np.arange(int(self.detection.window_size/6), self.detection.events.shape[1]) * self.detection.trace.sampling,
                 fitted_ev, c=self.orange_color, ls='--', label='fit')

        plt.ylabel(f'{self.detection.trace.y_unit}')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()

    def plot_singular_event_average(self):
        '''Plot event overlay + avg for events that have no overlapping events'''
        win_time = self.detection.window_size * self.detection.trace.sampling
        no_events_in_decay = np.where(np.diff(self.detection.event_peak_times) > win_time * 1.5)[0]
        no_events_in_rise = np.where(np.diff(self.detection.event_peak_times) > win_time * 0.5)[0] + 1
        intersection = np.intersect1d(no_events_in_rise, no_events_in_decay, assume_unique=False, return_indices=False)
        events = self.detection.events[intersection]
        # events = self.detection.events[no_events_in_decay]
        # events = self.detection.events[no_events_in_rise]
        fig = plt.figure(f'singular_events')
        plt.plot(events.T, c=self.main_trace_color, alpha=0.3)
        plt.plot(np.mean(events.T, axis=1), c=self.red_color, linewidth='3', label='average event')
        plt.show()



    def plot_event_histogram(self, plot: str='amplitude', cumulative: bool=False) -> None:
        ''' Plot event amplitude or frequency histogram '''
        if not self.detection.events_present():
            return
        if plot == 'frequency':
            data = np.diff(self.detection.event_locations * self.detection.trace.sampling, prepend=0)
            xlab_str = 'inter-event interval (s)'
        elif plot == 'amplitude':
            data = self.detection.event_stats.amplitudes
            xlab_str = f'amplitude ({self.detection.trace.y_unit})'
        else:
            return
        histtype = 'step' if cumulative else 'bar'
        ylab_str = 'cumulative frequency' if cumulative else 'count'
        fig = plt.figure(f'{plot}_histogram')
        plt.hist(data, bins='auto', cumulative=cumulative, density=cumulative, histtype=histtype, color=self.main_trace_color)
        plt.ylabel(ylab_str)
        plt.xlabel(xlab_str)
        plt.show()

    def plot_prediction(self, include_data: bool=False, plot_event_params: bool=False, plot_filtered_prediction: bool=False, 
                        plot_filtered_trace: bool=False, save_fig: str='') -> None:
        ''' 
        Plot prediction trace, optionally together with data and detection result.
        
        include_data: bool
            Boolean whether to include data and detected event peaks in the plot.
        plot_event_params: bool
            Boolean whether to plot event onset and half decay points.
        plot_filtered_prediction: bool
            Boolean whether to plot filtered prediction trace (maximum filter).
        plot_filtered_trace: bool
            Boolean whether to plot filtered prediction trace (hann window). If
            True, the first and last 100 points remain unchanged, to mask edge artifacts.
        save_fig: str
            Filename to save the figure to (in SVG format). If provided, plot will not be shown.
        '''
                
        fig = plt.figure('prediction')
        if include_data:
            ax1 = plt.subplot(211)
        prediction_x = np.arange(0, self.detection.prediction.shape[0]) * self.detection.trace.sampling
        if plot_filtered_prediction:
            plt.plot(prediction_x, maximum_filter1d(self.detection.prediction, size=int(5*self.detection.interpol_factor), origin=-2), c=self.main_trace_color)
        else:
            plt.plot(prediction_x, self.detection.prediction, c=self.main_trace_color)
        plt.axhline(self.detection.model_threshold, ls='--', c=self.orange_color)
        plt.ylabel('probability')

        if include_data:
            plt.tick_params('x', labelbottom=False)
            _ = plt.subplot(212, sharex=ax1)
            if plot_filtered_trace:
                main_trace = self.detection.hann_filter(self.detection.trace.data, filter_size=self.detection.convolve_win)
                plt.plot(self.detection.trace.time_axis, self.detection.trace.data, c='k', alpha=0.4)

            else:
                main_trace = self.detection.trace.data

            plt.plot(self.detection.trace.time_axis, main_trace, c=self.main_trace_color)
            
            try:
                plt.scatter(self.detection.event_peak_times, main_trace[self.detection.event_peak_locations], c=self.orange_color, s=20, zorder=2, label='peak positions')
                
                if plot_event_params:
                    plt.scatter(self.detection.event_start_times, main_trace[self.detection.event_start], c=self.red_color, s=20, zorder=2, label='event onset')
                    
                    ### remove np.nans from halfdecay
                    half_decay_for_plot = self.detection.half_decay[np.argwhere(~np.isnan(self.detection.half_decay)).flatten()].astype(np.int64)
                    half_decay_times_for_plot = self.detection.half_decay_times[np.argwhere(~np.isnan(self.detection.half_decay_times)).flatten()]
                    plt.scatter(half_decay_times_for_plot, main_trace[half_decay_for_plot], c=self.green_color, s=20, zorder=2, label='half decay')

                data_range = np.abs(np.max(main_trace) - np.min(main_trace))
                dat_min = np.min(main_trace)
                plt.eventplot(self.detection.event_peak_times, lineoffsets=dat_min - data_range/15, 
                              linelengths=data_range/20, color='k', lw=1.5)
            except:
                pass
            plt.tick_params('x')
            plt.ylabel(f'{self.detection.trace.y_unit}')
            
        plt.xlabel('time (s)')
        plt.legend(loc='upper right')
        if save_fig:
            if not save_fig.endswith('.svg'):
                save_fig = save_fig + '.svg'
            plt.savefig(save_fig, format='svg')
            plt.clf()
            plt.close()
            return
        plt.show()

    def plot_event_locations(self, plot_filtered: bool=False, save_fig: str='') -> None:
        ''' 
        Plot prediction trace, together with data and detected event positions (before any actual analysis is done).
        
        plot_filtered: bool
            Boolean whether to plot filtered prediction trace (maximum filter).
        save_fig: str
            Filename to save the figure to (in SVG format). If provided, plot will not be shown.
        '''
        fig = plt.figure('event locations')
        ax1 = plt.subplot(211)

        if plot_filtered:
            plt.plot(maximum_filter1d(self.detection.prediction, size=int(5*self.detection.interpol_factor), origin=-2), c=self.main_trace_color)
        else:
            plt.plot(self.detection.prediction, c=self.main_trace_color)

        plt.axhline(self.detection.model_threshold, color=self.orange_color, ls='--')
        plt.ylabel('probability')
        plt.tick_params('x', labelbottom=False)

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.detection.trace.data, c=self.main_trace_color)
        
        try:
            plt.scatter(self.detection.event_locations, self.detection.trace.data[self.detection.event_locations], c=self.orange_color, s=20, zorder=2)
            data_range = np.abs(np.max(self.detection.trace.data) - np.min(self.detection.trace.data))
            dat_min = np.min(self.detection.trace.data)
            plt.eventplot(self.detection.event_locations, lineoffsets=dat_min - data_range/15, 
                          linelengths=data_range/20, color='k', lw=1.5)
        except IndexError as e:
            pass
        
        plt.tick_params('x')
        plt.ylabel(f'{self.detection.trace.y_unit}')
        plt.xlabel('time in points')
        if save_fig:
            if not save_fig.endswith('.svg'):
                save_fig = save_fig + '.svg'
            plt.savefig(save_fig, format='svg')
            plt.close()
        else:
            plt.show()

    def plot_detection(self, save_fig: str='') -> None:
        ''' 
        Plot detection results together with data.
        
        save_fig: str
            Filename to save the figure to (in SVG format).
        '''
        fig = plt.figure('detection')
        plt.plot(self.detection.trace.time_axis, self.detection.trace.data, zorder=1, c=self.main_trace_color)
        if hasattr(self.detection, 'event_stats'):
            plt.scatter(self.detection.event_peak_times, self.detection.trace.data[self.detection.event_peak_locations], c=self.orange_color, s=20, zorder=2)
            dat_range = np.abs(np.max(self.detection.trace.data) - np.min(self.detection.trace.data))
            dat_min = np.min(self.detection.trace.data)
            plt.eventplot(self.detection.event_peak_times, lineoffsets=dat_min - dat_range/15, linelengths=dat_range/20, color='k', lw=1.5)

        plt.xlabel('s')
        plt.ylabel(f'{self.detection.trace.y_unit}')
        
        if save_fig:
            if not save_fig.endswith('.svg'):
                save_fig = save_fig + '.svg'
            plt.savefig(save_fig, format='svg')
            plt.close()
            return
        plt.show()
