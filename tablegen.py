# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:35:05 2020

@author: cjack
"""

import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy import interpolate

"""
Use single channel audio files for your signal input.

Make a Time variant (traditional) wavetable:
    

* Load a wav file with the following format:
 [file number]_[filename]_[frequency in hz]_.wav
 note the trailing underscore
* Ensure that the first part of the wavetable aligned with the start
of the wave form. I usually use a zero crossing here.
* put all of the files of interests in the input folder
* Run the function "run_tv"

~~~~~~~~~~~~~~~~

Make a Frequency variant wavetable:
    
* Load a wav file with the following format:
 [file number]_[filename].wav
* ensure that each waveform starts with a zero crossing and has similar behavior
* E.g. if the lowest note starts at 0 and then begins to increase,
 all following waveforms should start at a 0 crossing and then begin to increase
 
"""


# data importing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Data():
    
    def __init__(self, **kwargs):
        
        self.file = kwargs.get('file', '')
        self.freq = kwargs.get('freq', 1)
        self.samples = kwargs.get('samples', 1)
        self.maxsamples = kwargs.get('max_samples', 1)
        self.signal = kwargs.get('signal', np.zeros(1))
        self.order = kwargs.get('order', 0)

#--------------------------

def time_variant_data(inpath = "input", srate = 44100):
    
    files = list(filter(lambda x: '.wav' in x, os.listdir(inpath)))
    
    freqs = [float(x.split("_")[2]) for x in files]
    
    order = [float(x.split("_")[0]) for x in files]
    
    samples =[round(srate/f) for f in freqs]
    
    maxsamples = [max(samples) for x in samples]
    
    files = [os.path.join(inpath, file) for file in files]
    
    #0 ind is wavfile srate 1 ind is signal
    signals = [wavfile.read(file)[1] for file in files]
    
    temp = []
    
    for i in range(len(files)):
        temp.append((order[i], files[i], samples[i], maxsamples[i], signals[i]))
        
    temp = sorted(temp, key = lambda x: x[1])
    
    data = []
    for i in range(len(temp)):
        data.append(Data(order   = order[i], 
                         file   = files[i],
                         samples = samples[i],
                         maxsamples = maxsamples[i],
                         signal = signals[i])
                    )
    
    return data

#--------------------------

def single_cycle_data(inpath = 'input'):
    
    files = list(filter(lambda x: '.wav' in x, os.listdir(inpath)))
    
    order = [float(x.split("_")[0]) for x in files]
    
    files = [os.path.join(inpath, file) for file in files]
    
     #0 ind is wavfile srate 1 ind is signal
    signals = [wavfile.read(file)[1] for file in files]
    
    samples = [s.shape[0] for s in signals]
    
    temp = []
    
    for i in range(len(files)):
        temp.append((order[i], files[i], samples[i], signals[i]))
        
    temp = sorted(temp, key = lambda x: x[0])
    
    data = []
    
    for i in range(len(temp)):
        data.append(Data(order   = order[i], 
                         file   = files[i],
                         samples = samples[i],
                         signal = signals[i])
                    )
    return data
    

# windows ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_window(targetblock, center = 0.5, width = .33):
    X = targetblock
    mu = targetblock * center
    sigma = targetblock * width
    window = np.exp(-0.5*((np.arange(X)- mu)**2)/sigma**2)
    window = window - np.min(window)
    window = window/np.max(window)
    
    return window

#--------------------------

def sigmoid_window(targetblock, bounds = 7, pct = .05):
    size = int(targetblock * pct *0.5)
    x = np.linspace(-bounds, bounds, size*2)
    lsegment = 1/(1 + np.exp(-x))
    lsegment = lsegment - np.min(lsegment)
    lsegment = lsegment /np.max(lsegment)
    rsegment = np.flip(lsegment)
    
    window = np.ones(targetblock)
    window[:size * 2] = lsegment
    window[targetblock - size * 2: targetblock] = rsegment
    
    return window

#--------------------------

def linear_window(targetblock, pct = .01):
    size = int(targetblock * pct)
    lsegment = np.linspace(0,1, size)
    rsegment = np.flip(lsegment)
    
    window = np.ones(targetblock)
    window[:size] = lsegment
    window[targetblock - size: targetblock] = rsegment
    
    return window

    

# table creation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def time_variant_table(data, name = 'wavetable', targetblock = 2048, 
                     srate = 44100, limit = 256, vol = 29204):
    
    table = []
    window = linear_window(targetblock)
        
    t_interps = []
    
    for d in data:
        
        signal = d.signal
        loops = min(len(signal)/d.maxsamples, len(signal)/d.samples)
        loops = int(loops) - 1 
        # skip the last block with - 1
        # it may not be a full waveform
        
        x = np.linspace(0, targetblock, d.samples)
        for l in range(loops):
            start = d.samples * l
            end = start + d.samples
            
            y = signal[start:end]
            f = interpolate.interp1d(x,y)
            
            newblock = f(np.arange(0,targetblock))
            newblock = newblock * window

            
            t_interps.append(newblock)
    
    t_interps = np.array(t_interps)
    
    
    # frequency interpolation
    # expanding the table so it fills the entire limit
    
    f_interps = []
    f_in = np.arange(0,limit)
    for i in range(t_interps.shape[1]):
        
        x = np.linspace(0, limit - 1 ,t_interps.shape[0])
        y = t_interps[:,i]
        
        f = interpolate.interp1d(x,y)
        f_interps.append(f(f_in))
    
    
    table = np.array(f_interps).transpose()
    table = table * vol / np.max(np.abs(table))
    table = table.flatten().astype(np.int16)  
    
    wavfile.write(name+'.wav', srate, table)
    
    return table

def single_cycle_table(data, name = 'wavetable', targetblock = 2048,
                       srate = 44100, limit = 256, vol = 29204):
    
    window = linear_window(targetblock)
    
    # time interpolations
    # expanding the waveform so it fills the entire block
    t_interps = []
    f_in = np.arange(0,targetblock)
    for d in data:
        y = d.signal
        x = np.linspace(0,targetblock, d.samples)
        
        
        f = interpolate.interp1d(x, y, fill_value = 0)
        
        #newblock = f(f_in) * window
        
        #disabling windowing
        newblock = f(f_in)
        newblock = newblock * vol / np.max(np.abs(newblock))
        t_interps.append( newblock )
    
    t_interps = np.array(t_interps)
    
    

    # frequency interpolation
    # expanding the table so it fills the entire limit
    f_interps = []
    f_in = np.arange(0,limit)
    for i in range(t_interps.shape[1]):
        x = np.linspace(0, limit - 1 ,t_interps.shape[0])
        y = t_interps[:,i]
        
        f = interpolate.interp1d(x,y)
        
        f_interps.append(f(f_in))
    
    
    table = np.array(f_interps).transpose()

        
    
    #return table
    #table = table * vol / np.max(np.abs(table))
    table = table.flatten().astype(np.int16)
    
    
    wavfile.write(name+'.wav', srate, table)
    return  table
   
        
    
def run_sc(name = 'wavetable', srate = 44100, targetblock = 2048, limit = 256):
    data = single_cycle_data()
    table = single_cycle_table(data, name, targetblock, srate, limit)
    
# It's easier to just use built in wavetable imports.
def run_tv(name = 'wavetable', srate = 44100, targetblock = 2048, limit = 256):
    data = time_variant_data()
    table = time_variant_table(data, name, targetblock, srate, limit)
    
def run_sc_bulk(rootfolder, subpath = 'k', srate = 44100, targetblock = 2048, limit = 256):
    
    folders = os.listdir(rootfolder)
    
    for folder in folders:
        path = os.path.join(rootfolder, folder, subpath)
        if os.path.exists(path):
            print(path)
            try:
                data = single_cycle_data(path)
                table = single_cycle_table(data, folder, targetblock, limit)
            except Exception as e:
                print('error occured \n')
                
if __name__ == '__main__':
    run_sc()
                