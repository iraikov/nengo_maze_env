import numpy as np
import nengo_extras
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )
import matplotlib
import matplotlib.pyplot as plt


font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

tstop = 10.
dt = 0.001
t = np.linspace(0, tstop,  int(tstop/dt))
xy = np.load("xy.npy")
xy_reader = np.load("xy_reader.npy")
xy_reader_spikes = np.load("xy_reader_spikes.npy")


plt.figure(figsize=(16, 6))
rates = rates_kernel(t, xy_reader_spikes).T
im = plt.imshow(rates, origin='upper', aspect='auto', interpolation='none',
                extent=[np.min(t), np.max(t), 0, rates.shape[0]])
plt.show()

t_sample, xy_reader_spikes_sample = sample_by_variance(t, xy_reader_spikes, num=250, filter_width=0.1)
t_cycle_idxs = np.where(np.logical_and(t_sample >= 4.5, t_sample <= 10))[0]

plt.figure(figsize=(16, 6))
rates = rates_kernel(t_sample, xy_reader_spikes_sample, tau=0.1).T
peaks = np.argmax(rates, axis=1)
sorted_peaks = np.argsort(peaks)

plt.subplot(311)

plt.plot(t[t_cycle_idxs], xy[t_cycle_idxs,0], label="X")
plt.plot(t[t_cycle_idxs], xy[t_cycle_idxs,1], label="Y")
plt.legend()

plt.subplot(312)

plt.plot(t[t_cycle_idxs], xy_reader[t_cycle_idxs,1], label="decoded X (A.U.)")
plt.plot(t[t_cycle_idxs], xy_reader[t_cycle_idxs,0], label="decoded Y (A.U.)")
plt.legend()

plt.subplot(313)

im = plt.imshow(rates[:, t_cycle_idxs][sorted_peaks,:], origin='upper', aspect='auto', interpolation='none',
                extent=[np.min(t[t_cycle_idxs]), np.max(t[t_cycle_idxs]), 0, rates.shape[0]])
plt.show()

