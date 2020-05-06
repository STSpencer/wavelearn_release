'''Script to import sim_telarray GCT-S simulations,and export them to hdf5 files. Designed to mix
proton, gamma and now electron datafiles together for CTANN training/testing.
Needs ctapipe, remember to use source activate cta-dev in advance.
This version designed for use on the grid.
Written by S.T. Spencer (samuel.spencer@physics.ox.ac.uk) 8/8/2018'''


import ctapipe
from ctapipe.io.hessio import HESSIOEventSource
from time import sleep
from ctapipe.instrument import CameraGeometry
from ctapipe.calib import CameraCalibrator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
# from calibration_pipeline import display_telescope
from ctapipe.core import Tool
import numpy as np
import random
import time
import tables
import sys
import pyhessio
from traitlets import (Integer, Float, List, Dict, Unicode)
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.image.charge_extractors import SimpleIntegrator, AverageWfPeakIntegrator
from ctapipe.image.charge_extractors import LocalPeakIntegrator, GlobalPeakIntegrator, FullIntegrator
from traitlets import Int
import scipy.signal as signals
from scipy.interpolate import UnivariateSpline
import os
import signal
from scipy.interpolate import splrep, sproot, splev
import logging
import astropy.units as unit

from copy import deepcopy
from ctapipe.core import Component
from numba import jit


logging.basicConfig(level='warning')

class MultiplePeaks(Exception):
    pass

class NoPeaksFound(Exception):
    pass

def sig_handler(signum, frame):
    print("segfault")

def smooth(x, window_len=8, window='hanning'):
    """Smooth the data using a window with requested size.

r    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def findpeaks(teldata, cut_amp):
    '''Finds the peak location and amplitude in a waveform.'''
    pix_ids = np.arange(len(teldata))
    # Peak widths to search for (let's fix it at 8 samples, about the width of
    # the peak)
    widths = np.array([8, ])
    peaks = [signals.find_peaks_cwt(smooth(trace), widths)
             for trace in teldata[pix_ids]]
    peaks2 = np.zeros(2304, dtype=int)
    loclist = []
    for i in np.arange(len(peaks)):
        peaks[i] = peaks[i][np.where(np.asarray(peaks[i]) < 96)]
        for j in peaks[i]:
            # Perform peak amplitude cuts
            if smooth(teldata[i])[j] > cut_amp:
                peaks2[i] = j
                loclist.append(i)
                break
            else:
                continue

    peaks = np.reshape(peaks2, (48, 48))
    return peaks, loclist, pix_ids

@jit
def timemaker(X, cut_amp):
    '''Function to extract times of arrival from CHEC waveforms, makes 2d images of photon TOA.'''
    X = np.squeeze(X)
    Y = np.zeros((48, 48, 1))
    X = np.reshape(X, (2304, 96, 1))
    data = X[:, :, 0]
    data = data.reshape((2304, 96))
    peaks0 = findpeaks(data[:, :], cut_amp)
    Y[:, :, 0] = peaks0[0]
    return Y

@jit
def cam_squaremaker(data):
    '''Function to translate CHEC-S integrated images into square arrays for
    analysis purposes.'''
    square = np.zeros(2304)
    i = 0
    while i < 48:
        if i < 8:
            xmin = 48 * i + 8
            xmax = 48 * (i + 1) - 8
            square[xmin:xmax] = data[i * 32:(i + 1) * 32]
            i = i + 1
        elif i > 7 and i < 40:
            square[384:1920] = data[256:1792]
            i = 40
        else:
            xmin = 48 * i + 8
            xmax = 48 * (i + 1) - 8
            square[xmin:xmax] = data[512 + i * 32:544 + i * 32]
            i = i + 1

    square.resize((48, 48))
    square = np.flip(square, 0)
    return square

@jit
def cubemaker(data):
    '''Function to translate CHEC-S waveform datacubes into cube arrays for
    analysis purposes.'''
    square = np.zeros((2304, 96))
    i = 0
    while i < 48:
        if i < 8:
            xmin = 48 * i + 8
            xmax = 48 * (i + 1) - 8
            square[xmin:xmax, :] = data[i * 32:(i + 1) * 32, :]
            i = i + 1
        elif i > 7 and i < 40:
            square[384:1920, :] = data[256:1792, :]
            i = 40
        else:
            xmin = 48 * i + 8
            xmax = 48 * (i + 1) - 8
            square[xmin:xmax, :] = data[512 + i * 32:544 + i * 32, :]
            i = i + 1

    square.resize((48, 48, 96))
    square = np.flip(square, 0)
    return square

hists={}

@jit
def process_pedestal(event, output=False):
    '''Performs low level calibration of waveforms.'''
    chan=0
    for tel in event.dl0.tels_with_data:
        geom = event.inst.subarray.tel[tel].camera
        im = np.squeeze(event.dl1.tel[tel].image[chan])

        # Select pixels that are not signal and calculate
        # means and std of them (they should be pedestals):
        mask = tailcuts_clean(geom, im, picture_thresh=7, boundary_thresh=4)
        for ii in range(3):
            mask = dilate(geom, mask)

        if output:
            print('calib', event.dl0)

        hist, ed = np.histogram(im[~mask], bins=200, range=[-10, 10])

        if geom.cam_id in hists:
            hists[geom.cam_id][0] += hist
            hists[geom.cam_id][2] += len(im[~mask])  # counter
        else:
            hists[geom.cam_id] = [hist, ed, len(im[~mask])]
        return event

@jit
def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = np.amax(y) / 2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        return 0
    elif len(roots) < 2:
        return 0
    else:
        return abs(roots[1] - roots[0])

def rtft(x, peakloc):
    '''Calculates rise and fall times of pulses.'''
    peakloc = peakloc[0][0]
    diffs = np.diff(x)
    rarr = diffs[:peakloc]
    farr = diffs[peakloc:]
    rtab = np.where(rarr <= 0)[0]
    ftab = np.where(farr >= 0)[0]
    try:
        rtloc = rtab[-1]
        rt = peakloc - rtloc

    except IndexError:
        rtloc = 0
        rt = 0
    try:
        ftloc = (ftab)[0]
    except IndexError:
        ftloc = 0
    ft = ftloc
    return rt, ft

def main():
    signal.signal(signal.SIGSEGV, sig_handler)

    # Run Options
    # Import raw sim_telarray output files
    # TPA : problem here is that using:
    # python simtel_writer_dirac.py $SIMTEL_FILE in the bash script, there is no ordering in the simtel files...
    # a possible solution is to have them as generic, file1, file2 and file3, then check the first event and look at 
    # event.mc.shower_primary_id?
    gamma_data = sys.argv[1]
    hadron_data = sys.argv[2]
    electron_data = sys.argv[3]
    runcode = str(sys.argv[4])
    print(gamma_data,hadron_data,electron_data,runcode)

    output_filename = sys.argv[4]

    # Max number of events to read in for each of gammas/protons for training.
    event_nos = []
    gamma_source=HESSIOEventSource(input_url=gamma_data)
    gammacount = 0
    for event in gamma_source:
        gammacount+=1
    event_nos.append(gammacount)
    pyhessio.close_file()
    hadron_source=HESSIOEventSource(input_url=hadron_data)
    hadroncount = 0
    for event in hadron_source:
        hadroncount+=1
    event_nos.append(hadroncount)
    pyhessio.close_file()
    electron_source=HESSIOEventSource(input_url=electron_data)
    electroncount = 0
    for event in electron_source:
        electroncount+=1
    event_nos.append(electroncount)
    pyhessio.close_file()

    max2=min(event_nos)
    print(gammacount,hadroncount,electroncount)
    maxevents=max2
    no_tels = 4  # Number of telescopes
    cut_amp = 70 # Number of required counts for parameterization.

    count = 1  # Keeps track of number of events processed

    integrator = SimpleIntegrator(None, None)
    caliber = CameraCalibrator()

    # Read in gammas/ protons from simtel for each output file.
    
    # Basic principle is to load in, calibrate and parameterize the gamma ray
    # events, then do the same for the protons. Then mix the two together and
    # write them to disk.
        
    # Initialize lists for hdf5 storage.
    to_matlab = {
        'id': [],
        'event_id': [],
        'label': [],
        'mc_energy':[],
        'HasData': [],
        'tel_labels': [],
        'tel_data': [],
        'tel_integrated': [],
        'peak_times': [],
        'waveform_mean': [],
        'FWHM': [],
        'waveform_rms': [],
        'RT': [],
        'FT': [],
        'waveform_amplitude': []}

        
    gamma_source = HESSIOEventSource(input_url=gamma_data)
    
    # Determine events to load in using event seeker.
    
    for event in gamma_source:
        if count>=maxevents:
            break
        caliber.calibrate(event)
        event=process_pedestal(event)
        to_matlab['id'].append(count)
        to_matlab['event_id'].append(str(event.r0.event_id) + '01')
        to_matlab['label'].append(0.0)
        energy=event.mc.energy.to(unit.GeV)
        to_matlab['mc_energy'].append(energy.value)
        hasdata = np.zeros(4)
        for i in event.r0.tels_with_data:
            hasdata[i - 1] = 1
        to_matlab['HasData'].append(hasdata)
            
        # Initialize arrays for given event
        tel_labels = np.zeros((no_tels, 1),dtype='float32')
        datas = np.zeros((no_tels, 48, 48, 96),dtype='float32')
        integrated = np.zeros((no_tels, 48, 48),dtype='float32')
        timesarr = np.zeros((no_tels, 48, 48),dtype='float32')
        fwhmarr = np.zeros((no_tels, 48, 48),dtype='float32')
        rtarr = np.zeros((no_tels, 48, 48),dtype='float32')
        ftarr = np.zeros((no_tels, 48, 48),dtype='float32')
        meanarr = np.zeros((no_tels, 48, 48),dtype='float32')
        rmsarr = np.zeros((no_tels, 48, 48),dtype='float32')
        amparr = np.zeros((no_tels, 48, 48),dtype='float32')
        
        for index, tel_id in enumerate(event.dl0.tels_with_data):
            # Loop through all triggered telescopes.
            tel_label = 'tel_' + str(tel_id) + '_data'
            tel_labels[tel_id - 1, 0] = 0.0
            geom = event.inst.subarray.tel[tel_id].camera
            geom.make_rectangular()
            teldata = event.dl0.tel[tel_id]
            traces = teldata.waveform
            traces = traces[0]
            cubed = cubemaker(traces)
            datas[tel_id - 1, :, :, :] = cubed
            integ_charges = event.dl1.tel[tel_id]
            squared = cam_squaremaker(integ_charges['image'][0, :])
            integrated[tel_id - 1, :, :] = squared
            
            fwhmmat = []
            rtmat = []
            ftmat = []
            meanmat = []
            rmsmat = []
            ampmat = []

            ptimes = timemaker(cubed, cut_amp)
            timesarr[tel_id - 1, :, :] = ptimes[:, :, 0]
            
            for x in traces:
                x = smooth(x)
                # Perform peak amplitude cuts for parameterization.
                
                if np.amax(x) > cut_amp:
                    fwhmval = fwhm(np.arange(len(x)), x)
                    fwhmmat.append(fwhmval)
                    ampl = np.amax(x)
                    ampmat.append(ampl)
                    meanmat.append(np.mean(x))
                    rmsmat.append(np.sqrt(np.mean(np.square(x))))
                    rt, ft = rtft(x, np.where(x == ampl))
                    rtmat.append(rt)
                    ftmat.append(ft)
                else:
                    fwhmmat.append(0)
                    rtmat.append(0)
                    ftmat.append(0)
                    meanmat.append(0)
                    rmsmat.append(0)
                    ampmat.append(0)
                    
            # Make 2d histograms of parameters.
            fwhmarr[tel_id - 1, :,
                    :] = cam_squaremaker(np.asarray(fwhmmat))[:, :]
            rtarr[tel_id - 1, :, :] = cam_squaremaker(np.asarray(rtmat))[:, :]
            ftarr[tel_id - 1, :, :] = cam_squaremaker(np.asarray(ftmat))[:, :]
            meanarr[tel_id - 1, :,
                    :] = cam_squaremaker(np.asarray(meanmat))[:, :]
            rmsarr[tel_id - 1, :,
                   :] = cam_squaremaker(np.asarray(rmsmat))[:, :]
            amparr[tel_id - 1, :,
                   :] = cam_squaremaker(np.asarray(ampmat))[:, :]

        # Log telescopes that don't trigger.
        for i in np.arange(4):
            if hasdata[i] == 0:
                tel_labels[i] = -1
            else:
                tel_labels[i] = 0
                    
        # Send to hdf5 writer lists.
        # List of triggered telescopes
        to_matlab['tel_labels'].append(tel_labels)
        to_matlab['tel_data'].append(datas)
        to_matlab['tel_integrated'].append(integrated)
        to_matlab['peak_times'].append(timesarr)
        to_matlab['FWHM'].append(fwhmarr)
        to_matlab['RT'].append(rtarr)
        to_matlab['FT'].append(ftarr)
        to_matlab['waveform_mean'].append(meanarr)
        to_matlab['waveform_rms'].append(rmsarr)
        to_matlab['waveform_amplitude'].append(amparr)
            
        count = count + 1

    pyhessio.close_file()

    # Read in protons from simtel
    proton_hessfile = HESSIOEventSource(input_url=hadron_data)

    for event in proton_hessfile:
        if count>=2*maxevents:
            break
        caliber.calibrate(event)
        event = process_pedestal(event)
        to_matlab['id'].append(int(count))
        to_matlab['event_id'].append(str(event.r0.event_id) + '02')
        to_matlab['label'].append(1.0)
        energy=event.mc.energy.to(unit.GeV)
        to_matlab['mc_energy'].append(energy.value)
        hasdata = np.arange(4)
        
        for i in event.dl0.tels_with_data:
            hasdata[i - 1] = 1
            
        # Create arrays for event.
        to_matlab['HasData'].append(hasdata)
        tel_labels = np.zeros((no_tels, 1))
        datas = np.zeros((no_tels, 48, 48, 96),dtype='float32')
        integrated = np.zeros((no_tels, 48, 48),dtype='float32')
        timesarr = np.zeros((no_tels, 48, 48),dtype='float32')
        fwhmarr = np.zeros((no_tels, 48, 48),dtype='float32')
        rtarr = np.zeros((no_tels, 48, 48),dtype='float32')
        ftarr = np.zeros((no_tels, 48, 48),dtype='float32')
        meanarr = np.zeros((no_tels, 48, 48),dtype='float32')
        rmsarr = np.zeros((no_tels, 48, 48),dtype='float32')
        amparr = np.zeros((no_tels, 48, 48),dtype='float32')
        
        for index, tel_id in enumerate(event.dl0.tels_with_data):
            # Loop through triggered telescopes.
            tel_label = 'tel_' + str(tel_id) + '_data'
            tel_labels[tel_id - 1, 0] = 1.0
            geom = event.inst.subarray.tel[tel_id].camera
            geom.make_rectangular()
            teldata = event.dl0.tel[tel_id]
            traces = teldata.waveform
            traces=traces[0]
            cubed = cubemaker(traces)
            datas[tel_id - 1, :, :, :] = cubed
            integ_charges = event.dl1.tel[tel_id]
            squared = cam_squaremaker(integ_charges['image'][0, :])
            integrated[tel_id - 1, :, :] = squared
            ptimes = timemaker(cubed, cut_amp)
            timesarr[tel_id - 1, :, :] = ptimes[:, :, 0]
            fwhmmat = []
            rtmat = []
            ftmat = []
            meanmat = []
            rmsmat = []
            ampmat = []
                
            for x in traces:
                x = smooth(x)
                # Perform amplitude cut.
                if np.amax(x) > cut_amp:
                    fwhmval = fwhm(np.arange(len(x)), x)
                    fwhmmat.append(fwhmval)
                    ampl = np.amax(x)
                    ampmat.append(ampl)
                    meanmat.append(np.mean(x))
                    rmsmat.append(np.sqrt(np.mean(np.square(x))))
                    rt, ft = rtft(x, np.where(x == ampl))
                    rtmat.append(rt)
                    ftmat.append(ft)
                else:
                    fwhmmat.append(0)
                    rtmat.append(0)
                    ftmat.append(0)
                    meanmat.append(0)
                    rmsmat.append(0)
                    ampmat.append(0)
            
            # Make square histograms
            fwhmarr[tel_id - 1, :,
                    :] = cam_squaremaker(np.asarray(fwhmmat))[:, :]
            rtarr[tel_id - 1, :, :] = cam_squaremaker(np.asarray(rtmat))[:, :]
            ftarr[tel_id - 1, :, :] = cam_squaremaker(np.asarray(ftmat))[:, :]
            meanarr[tel_id - 1, :,
                    :] = cam_squaremaker(np.asarray(meanmat))[:, :]
            rmsarr[tel_id - 1, :,
                   :] = cam_squaremaker(np.asarray(rmsmat))[:, :]
            amparr[tel_id - 1, :,
                   :] = cam_squaremaker(np.asarray(ampmat))[:, :]
            
        for i in np.arange(4):
            if hasdata[i] == 0:
                tel_labels[i] = -1
            else:
                tel_labels[i] = 1
                
        # Send to hdf5 writer lists.
        # List of triggered telescopes
        to_matlab['tel_labels'].append(tel_labels)
        to_matlab['tel_data'].append(datas)
        to_matlab['tel_integrated'].append(integrated)
        to_matlab['peak_times'].append(timesarr)
        to_matlab['FWHM'].append(fwhmarr)
        to_matlab['RT'].append(rtarr)
        to_matlab['FT'].append(ftarr)
        to_matlab['waveform_mean'].append(meanarr)
        to_matlab['waveform_rms'].append(rmsarr)
        to_matlab['waveform_amplitude'].append(amparr)
        count = count + 1
        
    pyhessio.close_file()

    electron_hessfile = HESSIOEventSource(input_url=electron_data)

    for event in electron_hessfile:
        if count>=3*maxevents:
            break
        caliber.calibrate(event)
        event = process_pedestal(event)
        to_matlab['id'].append(int(count))
        to_matlab['event_id'].append(str(event.r0.event_id) + '03')
        to_matlab['label'].append(2.0)
        energy=event.mc.energy.to(unit.GeV)
        to_matlab['mc_energy'].append(energy.value)
        hasdata = np.arange(4)
        
        for i in event.dl0.tels_with_data:
            hasdata[i - 1] = 1
            
        # Create arrays for event.
        to_matlab['HasData'].append(hasdata)
        tel_labels = np.zeros((no_tels, 1))
        datas = np.zeros((no_tels, 48, 48, 96),dtype='float32')
        integrated = np.zeros((no_tels, 48, 48),dtype='float32')
        timesarr = np.zeros((no_tels, 48, 48),dtype='float32')
        fwhmarr = np.zeros((no_tels, 48, 48),dtype='float32')
        rtarr = np.zeros((no_tels, 48, 48),dtype='float32')
        ftarr = np.zeros((no_tels, 48, 48),dtype='float32')
        meanarr = np.zeros((no_tels, 48, 48),dtype='float32')
        rmsarr = np.zeros((no_tels, 48, 48),dtype='float32')
        amparr = np.zeros((no_tels, 48, 48),dtype='float32')
        
        for index, tel_id in enumerate(event.dl0.tels_with_data):
            # Loop through triggered telescopes.
            tel_label = 'tel_' + str(tel_id) + '_data'
            tel_labels[tel_id - 1, 0] = 2.0
            geom = event.inst.subarray.tel[tel_id].camera
            geom.make_rectangular()
            teldata = event.dl0.tel[tel_id]
            traces = teldata.waveform
            traces=traces[0]
            cubed = cubemaker(traces)
            datas[tel_id - 1, :, :, :] = cubed
            integ_charges = event.dl1.tel[tel_id]
            squared = cam_squaremaker(integ_charges['image'][0, :])
            integrated[tel_id - 1, :, :] = squared
            ptimes = timemaker(cubed, cut_amp)
            timesarr[tel_id - 1, :, :] = ptimes[:, :, 0]
            fwhmmat = []
            rtmat = []
            ftmat = []
            meanmat = []
            rmsmat = []
            ampmat = []
            
            for x in traces:
                x = smooth(x)
                # Perform amplitude cut.
                if np.amax(x) > cut_amp:
                    fwhmval = fwhm(np.arange(len(x)), x)
                    fwhmmat.append(fwhmval)
                    ampl = np.amax(x)
                    ampmat.append(ampl)
                    meanmat.append(np.mean(x))
                    rmsmat.append(np.sqrt(np.mean(np.square(x))))
                    rt, ft = rtft(x, np.where(x == ampl))
                    rtmat.append(rt)
                    ftmat.append(ft)
                else:
                    fwhmmat.append(0)
                    rtmat.append(0)
                    ftmat.append(0)
                    meanmat.append(0)
                    rmsmat.append(0)
                    ampmat.append(0)
                        
            # Make square histograms
            fwhmarr[tel_id - 1, :,
                    :] = cam_squaremaker(np.asarray(fwhmmat))[:, :]
            rtarr[tel_id - 1, :, :] = cam_squaremaker(np.asarray(rtmat))[:, :]
            ftarr[tel_id - 1, :, :] = cam_squaremaker(np.asarray(ftmat))[:, :]
            meanarr[tel_id - 1, :,
                    :] = cam_squaremaker(np.asarray(meanmat))[:, :]
            rmsarr[tel_id - 1, :,
                   :] = cam_squaremaker(np.asarray(rmsmat))[:, :]
            amparr[tel_id - 1, :,
                   :] = cam_squaremaker(np.asarray(ampmat))[:, :]
            
        for i in np.arange(4):
            if hasdata[i] == 0:
                tel_labels[i] = -1
            else:
                tel_labels[i] = 2
                        
        # Send to hdf5 writer lists.
        # List of triggered telescopes
        to_matlab['tel_labels'].append(tel_labels)
        to_matlab['tel_data'].append(datas)
        to_matlab['tel_integrated'].append(integrated)
        to_matlab['peak_times'].append(timesarr)
        to_matlab['FWHM'].append(fwhmarr)
        to_matlab['RT'].append(rtarr)
        to_matlab['FT'].append(ftarr)
        to_matlab['waveform_mean'].append(meanarr)
        to_matlab['waveform_rms'].append(rmsarr)
        to_matlab['waveform_amplitude'].append(amparr)
        
        count = count + 1

    pyhessio.close_file()

    # Make everything arrays in order to randomize.
    to_matlab['id'] = np.asarray(to_matlab['id'])
    to_matlab['event_id'] = np.asarray(to_matlab['event_id'])
    to_matlab['label'] = np.asarray(to_matlab['label'])
    to_matlab['mc_energy'] = np.asarray(to_matlab['mc_energy'])
    to_matlab['HasData'] = np.asarray(to_matlab['HasData'])
    to_matlab['tel_labels'] = np.asarray(to_matlab['tel_labels'])
    to_matlab['tel_integrated'] = np.asarray(to_matlab['tel_integrated'],dtype='float32')
    to_matlab['peak_times'] = np.asarray(to_matlab['peak_times'],dtype='float32')
    to_matlab['FWHM'] = np.asarray(to_matlab['FWHM'],dtype='float32')
    to_matlab['RT'] = np.asarray(to_matlab['RT'],dtype='float32')
    to_matlab['FT'] = np.asarray(to_matlab['FT'],dtype='float32')
    to_matlab['waveform_mean'] = np.asarray(to_matlab['waveform_mean'],dtype='float32')
    to_matlab['waveform_rms'] = np.asarray(to_matlab['waveform_rms'],dtype='float32')
    to_matlab['waveform_amplitude'] = np.asarray(
        to_matlab['waveform_amplitude'],dtype='float32')
        
    no_events = len(to_matlab['label'])
    randomize = np.arange(len(to_matlab['label']), dtype='int')
    
    # Implement uniform randomization here
    np.random.shuffle(randomize)
    to_matlab['id'] = to_matlab['id'][randomize]
    to_matlab['event_id'] = to_matlab['event_id'][randomize]
    to_matlab['label'] = to_matlab['label'][randomize]
    to_matlab['mc_energy'] = to_matlab['mc_energy'][randomize]
    to_matlab['HasData'] = to_matlab['HasData'][randomize]
    to_matlab['tel_labels'] = to_matlab['tel_labels'][randomize]
    to_matlab['tel_integrated'] = to_matlab['tel_integrated'][randomize]
    to_matlab['peak_times'] = to_matlab['peak_times'][randomize]
    to_matlab['FWHM'] = to_matlab['FWHM'][randomize]
    to_matlab['RT'] = to_matlab['RT'][randomize]
    to_matlab['FT'] = to_matlab['FT'][randomize]
    to_matlab['waveform_mean'] = to_matlab['waveform_mean'][randomize]
    to_matlab['waveform_rms'] = to_matlab['waveform_rms'][randomize]
    to_matlab['waveform_amplitude'] = to_matlab['waveform_amplitude'][randomize]
    
    h5file = tables.open_file(runcode+'.hdf5', mode="w")
    root = h5file.root
    print('Writing')
    
    # HDF5 writer code
    lab_event = h5file.create_array(root,
                                    'event_label', np.int32(to_matlab['label']),'event_label')
    id_group = h5file.create_array(root,
                                    'id', np.int32(to_matlab['id']),'id')
    id_group2 = h5file.create_array(root,
                                    'event_id', np.int32(to_matlab['event_id']),'event_id')
    energy_group = h5file.create_array(root,
                                    'mc_energy', np.float32(to_matlab['mc_energy']),'mc_energy')
    squared_group = h5file.create_array(root,
                                    'squared_training', np.float32(to_matlab['tel_integrated']),'event_label')
    labels_group = h5file.create_array(root,
                                    'tel_labels', np.int32(to_matlab['tel_labels']),'tel_labels')
    times_group = h5file.create_array(root,
                                    'peak_times', np.float32(to_matlab['peak_times']),'peak_times')

    fwhm_group = h5file.create_array(root,
                                    'FWHM', np.float32(to_matlab['FWHM']),'FWHM')
    rt_group = h5file.create_array(root,
                                    'RT', np.float32(to_matlab['RT']),'RT')
    ft_group = h5file.create_array(root,
                                    'FT', np.float32(to_matlab['FT']),'FT')
    mean_group = h5file.create_array(root,
                                    'waveform_mean', np.float32(to_matlab['waveform_mean']),'waveform_mean')
    rms_group = h5file.create_array(root,
                                    'waveform_rms', np.float32(to_matlab['waveform_rms']),'waveform_rms')
    amp_group = h5file.create_array(root,
                                    'waveform_amplitude', np.float32(to_matlab['waveform_amplitude']),'waveform_amplitude')
        
    h5file.close()


if __name__ == '__main__':
    main()
