import struct, os, sys, math
import numpy as np
import h5py

def h5_get_group(h, groupname):
    if groupname in h:
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g


def h5_get_dataset(g, dsetname, **kwargs):
    if dsetname in g:
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset

def export_dvs_events_to_h5(label, input_file_path, output_file_path):
    # ts x y p
    ee_ts = []
    ee_xs = []
    ee_ys = []
    ee_ps = []
    
    with open(input_file_path) as fp:
        lines = fp.readlines()
        for l in lines:
            a = l.split(' ')
            ee_t = float(a[0])
            ee_x = int(a[1])
            ee_y = int(a[2])
            ee_p = int(a[3])
            ee_ts.append(ee_t)
            ee_xs.append(ee_x)
            ee_ys.append(ee_y)
            ee_ps.append(ee_p)

    ee_ts = np.asarray(ee_ts, dtype=np.float32)        
    ee_xs = np.asarray(ee_xs, dtype=np.int32)        
    ee_ys = np.asarray(ee_ys, dtype=np.int32)        
    ee_ps = np.asarray(ee_ps, dtype=np.int8)

    h5_group = label
    with h5py.File(output_file_path, "a") as h5:
        g = h5_get_group(h5, h5_group)
        
        dset_ts = h5_get_dataset(g, 'ee_t', maxshape=(len(ee_ts),), dtype=np.float32)
        dset_ts.resize((len(ee_ts),))
        dset_ts[:] = ee_ts
        
        dset_xs = h5_get_dataset(g, 'ee_x', maxshape=(len(ee_xs),), dtype=np.int32)
        dset_xs.resize((len(ee_xs),))
        dset_xs[:] = ee_xs
        
        dset_ys = h5_get_dataset(g, 'ee_y', maxshape=(len(ee_ys),), dtype=np.int32)
        dset_ys.resize((len(ee_ys),))
        dset_ys[:] = ee_ys
    
        dset_ps = h5_get_dataset(g, 'ee_p', maxshape=(len(ee_ps),), dtype=np.int8)
        dset_ps.resize((len(ee_ps),))
        dset_ps[:] = ee_ps
    h5.close()     
     


def gen_dvs_rate_frames(T, X, Y, Pol, num_frames, width=240, height=180, decay=0.02):
    
    T = T.reshape((-1, 1))
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    Pol = Pol.reshape((-1, 1))

    time_step = math.ceil((T[-1]-T[0])/(num_frames+1))
    print(f"T[0] = {T[0]} T[-1] = {T[-1]} time_step = {time_step}")

    start_idx = 0
    end_idx = 0
    start_time = T[0]
    end_time = start_time + time_step

    frames = []
    start_timestamp = start_time*np.zeros((width*height, ), dtype=np.float32)
    start_rate = np.zeros((width*height, ), dtype=np.float32)
    
    while end_time <= T[-1] and len(frames) < num_frames: 
        while T[end_idx] < end_time:
            end_idx = end_idx + 1
        
        print(f"start_timestamp.shape = {start_timestamp.shape}")
        print(f"start_rate.shape = {start_rate.shape}")

        data_x = np.array(X[start_idx:end_idx]).reshape((-1, 1))
        data_y = np.array(Y[start_idx:end_idx]).reshape((-1, 1))
        data_t = np.array(T[start_idx:end_idx]).reshape((-1, 1))
        data_pol = np.array(Pol[start_idx:end_idx]).reshape((-1, 1))

        frame_len = end_idx-start_idx
        print(f"frame_len = {frame_len}")
        
        timestamp = np.tile(start_timestamp.reshape((-1, 1)), (1, frame_len))
        rate = np.tile(start_rate.reshape((-1, 1)), (1, frame_len))

        print(f"timestamp.shape = {timestamp.shape}")
        print(f"rate.shape = {rate.shape}")
        for i in range(1, data_x.shape[0]):
            data_idx = height * data_x[i] + data_y[i]
            #print(f"data_idx = {data_idx} data_x[{i}] = {data_x[i]} data_y[{i}] = {data_y[i]}")
            timestamp[data_idx, i] = data_t[i]
            time_interval = timestamp[data_idx, i] - timestamp[data_idx, i-1]
            np.maximum(rate[:, i-1] - decay*time_interval, 0., out=rate[:, i])
            if data_pol[i] == 1:
                rate[data_idx, i] += 1. * time_interval
           
        #frame = np.flip(255*(timestamp-start_time)/step_time, 0).astype(np.uint8)
        frame = np.mean(rate, axis=1)
        frames.append(frame)
        print(end_time)

        start_time = end_time
        end_time += time_step
        start_idx = end_idx
        start_timestamp = timestamp[:, -1]
        start_rate = rate[:, -1]

        
    return frames

