import librosa
from librosa import display
import time
import numpy as np
import os
import scipy.signal
from librosa import core, decompose, feature, util
from librosa.util.exceptions import ParameterError

# Dynamic Value Change
def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=1.5,high=3)
    return (data * dyn_change)

# White Noise Adding
def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data

def _signal_to_frame_nonsilent(y, frame_length=2048, hop_length=512, top_db=60, ref=np.max):
    # Convert to mono
    y_mono = core.to_mono(y)

    # Compute the MSE for the signal
    mse = feature.rms(y=y_mono,
                      frame_length=frame_length,
                      hop_length=hop_length)**2

    return (core.power_to_db(mse.squeeze(),
                             ref=ref,
                             top_db=None) > - top_db)

def trim(y, top_db=30, ref=np.max, frame_length=1024, hop_length=512):
    non_silent = _signal_to_frame_nonsilent(y,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            ref=ref,
                                            top_db=top_db)
    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(core.frames_to_samples(nonzero[0], hop_length))
        end = min(y.shape[-1],
                  int(core.frames_to_samples(nonzero[-1] + 1, hop_length)))
    else:
        # The signal only contains zeros
        start, end = 0, 0

    # Build the mono/stereo index
    full_index = [slice(None)] * y.ndim
    full_index[-1] = slice(start, end)

    #return y[tuple(full_index)], np.asarray([start, end])
    return y[start:end]

def split(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):

    non_silent = _signal_to_frame_nonsilent(y,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            ref=ref,
                                            top_db=top_db)

    edges = np.flatnonzero(np.diff(non_silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    # If the first frame had high energy, count it
    if non_silent[0]:
        edges.insert(0, [0])

    # Likewise for the last frame
    if non_silent[-1]:
        edges.append([len(non_silent)])

    # Convert from frames to samples
    edges = core.frames_to_samples(np.concatenate(edges),
                                   hop_length=hop_length)

    # Clip to the signal duration
    edges = np.minimum(edges, y.shape[-1])

    # Stack the results back as an ndarray
    return edges.reshape((-1, 2))


def data_augmentation(path, data_dynchange=False, data_noise=False, data_trim=True):
    lst = []
    start_time = time.time()
    print("--- Start load Data. Start time: %s ---" % (start_time))
    ########
    for subdir, dirs, files in os.walk(path):

        for file in files:
            try:
                X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
                sample_rate = np.array(sample_rate)
                
                if any([data_dynchange, data_noise, data_trim]):
                    pass
                else:
                    X_original = X
                    mfccs = np.mean(librosa.feature.mfcc(y=X_original, sr=sample_rate, n_mfcc=100).T,axis=0) 
                    file0 = int(file[7:8]) - 1 
                    arr_original = mfccs, file0
                    lst.append(arr_original)      
                
                # Data dynamic value change
                if bool(data_dynchange) :
                    X_dyn_change = dyn_change(X)
                    mfccs = np.mean(librosa.feature.mfcc(y=X_dyn_change, sr=sample_rate, n_mfcc=100).T,axis=0) 
                    file1 = int(file[7:8]) - 1 
                    arr_dyn_change = mfccs, file1
                    lst.append(arr_dyn_change)      
                else:
                    pass
                
                # Data noise adding
                if bool(data_noise) :            
                    X_noise = noise(X)
                    mfccs = np.mean(librosa.feature.mfcc(y=X_noise, sr=sample_rate, n_mfcc=100).T,axis=0) 
                    file2 = int(file[7:8]) - 1 
                    arr_noise = mfccs, file2
                    lst.append(arr_noise)   
                else:
                    pass
                
                # Data trimming
                if bool(data_trim):            
                    X_trim = trim(X)
                    mfccs = np.mean(librosa.feature.mfcc(y=X_trim, sr=sample_rate, n_mfcc=100).T,axis=0) 
                    file3 = int(file[7:8]) - 1 
                    arr_trim = mfccs, file3
                    lst.append(arr_trim)  
                else:
                    pass
                          
            # If the file is not valid, skip it
            except ValueError:
                continue
    ########
    print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
    return lst
