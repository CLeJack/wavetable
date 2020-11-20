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
import random

class Data():
    
    def __init__(self, file, freq, samples, maxsamples, signal):
        
        self.file = file
        self.freq = freq
        self.samples = samples
        self.maxsamples = maxsamples
        self.signal = signal

def wav_data(inpath = "input", srate = 44100):
    
    files = list(filter(lambda x: '.wav' in x, os.listdir(inpath)))
    
    freqs = [float(x.split("_")[1]) for x in files]
    
    samples =[round(srate/f) for f in freqs]
    
    maxsamples = [max(samples) for x in samples]
    
    files = [os.path.join(inpath, file) for file in files]
    
    #0 ind is wavfile srate 1 ind is signal
    signals = [wavfile.read(file)[1] for file in files]
    
    temp = []
    
    for i in range(len(files)):
        temp.append((files[i], freqs[i], samples[i], maxsamples[i], signals[i]))
        
    temp = sorted(temp, key = lambda x: x[1])
    
    data = []
    for i in range(len(temp)):
        data.append(Data(*temp[i]))
    
    return data

def get_window(targetblock, center = 0.5, width = .25):
    X = targetblock
    mu = targetblock * center
    sigma = targetblock * width
    window = np.exp(-0.5*((np.arange(X)- mu)**2)/sigma**2)
    window = window - np.min(window)
    window = window/np.max(window)
    
    return window

def create_wavetable(data, name = 'wavetable', targetblock = 2048, srate = 44100, limit = 256):
    
    table = []
    window = get_window(targetblock)
    
        
        
    for d in data:
        signal = d.signal
        loops = min(len(signal)/d.maxsamples, len(signal)/d.samples)
        loops = int(loops)
        
        x = np.linspace(0, targetblock, d.samples)
        for l in range(loops):
            start = d.samples* l
            end = start + d.samples
            y = signal[start:end]
            f = interpolate.interp1d(x,y)
            newblock = f(np.arange(0,targetblock))
            newblock = newblock * window
            
            table.append(newblock)
    
    table = np.array(table)
    selection = np.zeros(len(table))
    selection[:limit] = 1
    random.shuffle(selection)
    
    selection = selection == 1
    
    output = table[selection].flatten().astype(np.int16)
    
    wavfile.write(name+'.wav', srate, output)
    
    
    return output
            
    
def run(name = 'wavetable', srate = 44100, targetblock = 2048, limit = 256):
    data = wav_data(srate = srate)
    table = create_wavetable(data, name, srate, targetblock, srate, limit)
    