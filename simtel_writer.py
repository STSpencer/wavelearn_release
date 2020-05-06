'''Script to import sim_telarray GCT-S simulations,
optionally plot them, and export them to hdf5 files. Designed to mix
proton, gamma and now electron datafiles together for CTANN training/testing.
Needs ctapipe, remember to use source activate cta-dev in advance.
This version includes randomization of the two datasets
 and the storing of parameterized waveforms.
Written by S.T. Spencer (samuel.spencer@physics.ox.ac.uk) 8/8/2018'''

import time
import matplotlib as mpl
import ctapipe
from ctapipe.io.hessio import HESSIOEventSource, hessio_event_source
from matplotlib.colors import LogNorm
from time import sleep
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.calib import CameraCalibrator
from ctapipe.calib.camera import CameraDL0Reducer
from ctapipe.core import Tool
mpl.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as anim
import time
from scipy.io import savemat
import sys
import pyhessio
from astropy.io import fits
import h5py
from traitlets import (Integer, Float, List, Dict, Unicode)
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.image.charge_extractors import SimpleIntegrator, AverageWfPeakIntegrator
from ctapipe.image.charge_extractors import LocalPeakIntegrator, GlobalPeakIntegrator, FullIntegrator
from traitlets import Int
import scipy.signal as signals
from ctapipe.io import EventSeeker
from scipy.interpolate import UnivariateSpline
import os
import signal
from scipy.interpolate import splrep, sproot, splev
import logging
import astropy.units as unit
import numba
from numba import jit

logging.basicConfig(level='warning')

class MultiplePeaks(Exception):
    pass

class NoPeaksFound(Exception):
    pass

def sig_handler(signum, frame):
    print("segfault")

signal.signal(signal.SIGSEGV, sig_handler)

# Run Options
# Import raw sim_telarray output files
runno = 30
gamma_data = "/store/samsims/new/gamma/run" + str(runno) + ".simtel.gz"
hadron_data = "/store/samsims/new/proton3/run"+str(runno)+".simtel.gz"
electron_data = "/store/samsims/new/electron/run"+str(runno)+".simtel.gz"
gammaflag = 3  # Should be 1 to plot gammas or 2 for hadrons or 3 for electrons.
plot = False  # Whether or not to make animation plots for one single event.
event_plot = 0  # Min event number to plot
chan = 0  # PM Channel to use.
output_filename = '/store/spencers/Data/timetest'  # HDF5 files output name.

# Max number of events to read in for each of gammas/protons for training.
maxcount = 10
no_files = 1  # Number of files in which to store events
filerat = maxcount / no_files

print('Filerat', filerat)
no_tels = 4  # Number of telescopes
plot_wf = False  # Whether or not to plot the parameterized waveforms.
cut_amp = 70 # Number of required counts for parameterization.

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


def plot_cubed(cubed_list):
    '''Makes animations of waveforms for the purpose of inspecting data.'''
    fig3 = plt.figure()
    cubed_0 = cubed_list[0]
    cubed_0_mean = np.mean(cubed_0[:, :, :5])
    cubed_1 = cubed_list[1]
    cubed_1_mean = np.mean(cubed_1[:, :, :5])
    cubed_2 = cubed_list[2]
    cubed_2_mean = np.mean(cubed_2[:, :, :5])
    cubed_3 = cubed_list[3]
    cubed_3_mean = np.mean(cubed_3[:, :, :5])

    plt.subplot(221)
    im0 = plt.imshow(cubed_0[:, :, 0])
    plt.subplot(223)
    im1 = plt.imshow(cubed_1[:, :, 0])
    plt.subplot(222)
    im2 = plt.imshow(cubed_2[:, :, 0])
    plt.subplot(224)
    im3 = plt.imshow(cubed_3[:, :, 0])

    def updatefig(j):
        im0.set_array(cubed_0[:, :, j])
        im1.set_array(cubed_1[:, :, j])
        im2.set_array(cubed_2[:, :, j])
        im3.set_array(cubed_3[:, :, j])

        fig3.suptitle('T=' + str(j) + ' ns')
        return im0, im1, im2, im3

    ani = anim.FuncAnimation(fig3, updatefig, frames=range(96),
                             interval=3)

    ani.save(filename='/home/spencers/anim.mp4')
    plt.show()

    return 1


cmaps = [plt.cm.jet, plt.cm.winter,
         plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
         plt.cm.cool, plt.cm.coolwarm]

hists = {}
chan = 0  # which channel to look at

@jit
def process_pedestal(event, output=False):
    '''Performs low level calibration of waveforms.'''

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


count = 1  # Keeps track of number of events processed


integrator = SimpleIntegrator(None, None)
caliber = CameraCalibrator()

# Read in gammas/ protons from simtel for each output file.
starttime=time.time()
for fileno in np.arange(1, no_files + 1):

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
    smallfile_name = output_filename + "run" + str(runno) + "_" + str(fileno) + '.hdf5'

    gamma_source = HESSIOEventSource(input_url=gamma_data)

    print('Processing Gammas')
    evfinder = EventSeeker(reader=gamma_source)
    # Determine events to load in using event seeker.
    startev = int(filerat * fileno - filerat)
    midev = int(filerat * fileno - filerat / 2.0)
    endev = int(filerat * fileno)
    print(startev, endev)

    for event in evfinder[startev:endev]:
        caliber.calibrate(event)
        event = process_pedestal(event)

        if count % 1000 == 0:
            print(count)

        if plot == True and gammaflag == 2:
            break

        to_matlab['id'].append(count)
        to_matlab['event_id'].append(str(event.r0.event_id) + '01')
        to_matlab['label'].append(0.0)
        energy=event.mc.energy.to(unit.GeV)
        print(energy.value)
        to_matlab['mc_energy'].append(energy.value)
        hasdata = np.zeros(4)
        for i in event.r0.tels_with_data:
            hasdata[i - 1] = 1
        to_matlab['HasData'].append(hasdata)

        # Initialize arrays for given event
        tel_labels = np.zeros((no_tels, 1))
        datas = np.zeros((no_tels, 48, 48, 96))
        integrated = np.zeros((no_tels, 48, 48))
        timesarr = np.zeros((no_tels, 48, 48))
        fwhmarr = np.zeros((no_tels, 48, 48))
        rtarr = np.zeros((no_tels, 48, 48))
        ftarr = np.zeros((no_tels, 48, 48))
        meanarr = np.zeros((no_tels, 48, 48))
        rmsarr = np.zeros((no_tels, 48, 48))
        amparr = np.zeros((no_tels, 48, 48))
        cubed_list = []  # For plotting code

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

            if plot_wf == True:
                # Code to plot waveform parameters.
                plt.subplot(8, 1, 1)
                times = squared
                plt.imshow(times)
                plt.title('Charge')
                plt.axis('off')
                plt.subplot(8, 1, 2)
                times = cam_squaremaker(meanmat)
                plt.imshow(times)
                plt.title('Waveform Mean')
                plt.axis('off')
                plt.subplot(8, 1, 3)
                times = cam_squaremaker(ampmat)
                plt.imshow(times)
                plt.title('Pulse Amplitude')
                plt.axis('off')

                plt.subplot(8, 1, 4)
                times = ptimes
                plt.imshow(times[:, :, 0])
                plt.title('Pulse Time')
                plt.axis('off')

                plt.subplot(8, 1, 5)
                times = cam_squaremaker(rmsmat)
                plt.imshow(times)
                plt.title('Waveform RMS')
                plt.axis('off')

                plt.subplot(8, 1, 6)
                times = cam_squaremaker(np.asarray(fwhmmat))
                plt.imshow(times)
                plt.title('FWHM')
                plt.axis('off')

                plt.subplot(8, 1, 7)
                times = cam_squaremaker(np.asarray(rtmat))
                plt.imshow(times)
                plt.title('RT')
                plt.axis('off')

                plt.subplot(8, 1, 8)
                times = cam_squaremaker(np.asarray(ftmat))
                plt.imshow(times)
                plt.title('FT')
                plt.subplots_adjust(hspace=0.5)
                plt.axis('off')
                plt.show()

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

            if plot == True and count > event_plot and len(
                    event.r0.tels_with_data) > 3 and gammaflag == 1:
                cubed_list.append(cubed)
            else:
                continue

        # Log telescopes that don't trigger.
        for i in np.arange(4):
            if hasdata[i] == 0:
                tel_labels[i] = -1

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

        if plot == True and count > event_plot and len(
                event.r0.tels_with_data) > 3 and gammaflag == 1:
            plot_cubed(cubed_list)
            break

        count = count + 1

    pyhessio.close_file()

    # Read in protons from simtel
    print('Processing Protons')

    proton_hessfile = HESSIOEventSource(input_url=hadron_data)
    evfinder = EventSeeker(proton_hessfile)
    print(startev, endev)

    for event in evfinder[startev:endev]:
        caliber = CameraCalibrator()
        caliber.calibrate(event)
        event = process_pedestal(event)
        if count % 1000 == 0:
            print(count)
        to_matlab['id'].append(int(count))
        to_matlab['event_id'].append(str(event.r0.event_id) + '02')
        to_matlab['label'].append(1.0)
        energy=event.mc.energy.to(unit.GeV)
        print(energy.value)
        to_matlab['mc_energy'].append(energy.value)
        hasdata = np.arange(4)

        for i in event.dl0.tels_with_data:
            hasdata[i - 1] = 1

        # Create arrays for event.
        to_matlab['HasData'].append(hasdata)
        tel_labels = np.zeros((no_tels, 1))
        datas = np.zeros((no_tels, 48, 48, 96))
        integrated = np.zeros((no_tels, 48, 48))
        timesarr = np.zeros((no_tels, 48, 48))
        fwhmarr = np.zeros((no_tels, 48, 48))
        rtarr = np.zeros((no_tels, 48, 48))
        ftarr = np.zeros((no_tels, 48, 48))
        meanarr = np.zeros((no_tels, 48, 48))
        rmsarr = np.zeros((no_tels, 48, 48))
        amparr = np.zeros((no_tels, 48, 48))
        cubed_list = []  # For plotting code

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

            if plot_wf == True:
                # Code to plot waveform parameters.
                plt.subplot(8, 1, 1)
                times = squared
                plt.imshow(times)
                plt.title('Charge')
                plt.axis('off')
                plt.subplot(8, 1, 2)
                times = cam_squaremaker(meanmat)
                plt.imshow(times)
                plt.title('Waveform Mean')
                plt.axis('off')
                plt.subplot(8, 1, 3)
                times = cam_squaremaker(ampmat)
                plt.imshow(times)
                plt.title('Pulse Amplitude')
                plt.axis('off')

                plt.subplot(8, 1, 4)
                times = ptimes
                plt.imshow(times[:, :, 0])
                plt.title('Pulse Time')
                plt.axis('off')

                plt.subplot(8, 1, 5)
                times = cam_squaremaker(rmsmat)
                plt.imshow(times)
                plt.title('Waveform RMS')
                plt.axis('off')

                plt.subplot(8, 1, 6)
                times = cam_squaremaker(np.asarray(fwhmmat))
                plt.imshow(times)
                plt.title('FWHM')
                plt.axis('off')

                plt.subplot(8, 1, 7)
                times = cam_squaremaker(np.asarray(rtmat))
                plt.imshow(times)
                plt.title('RT')
                plt.axis('off')

                plt.subplot(8, 1, 8)
                times = cam_squaremaker(np.asarray(ftmat))
                plt.imshow(times)
                plt.title('FT')
                plt.subplots_adjust(hspace=0.5)
                plt.axis('off')
                plt.show()

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

            if plot == True and count > event_plot and len(
                    event.r0.tels_with_data) > 3 and gammaflag == 3:
                cubed_list.append(cubed)
            else:
                continue

        for i in np.arange(4):
            if hasdata[i] == 0:
                tel_labels[i] = -1

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

        if plot == True and count > event_plot and len(
                event.r0.tels_with_data) > 3 and gammaflag == 3:
            plot_cubed(cubed_list)
            break

        count = count + 1

    pyhessio.close_file()

    print('Processing Electrons')

    electron_hessfile = HESSIOEventSource(input_url=electron_data)
    evfinder = EventSeeker(electron_hessfile)
    print(startev, endev)

    for event in evfinder[startev:endev]:
        caliber = CameraCalibrator()
        caliber.calibrate(event)
        event = process_pedestal(event)
        if count % 1000 == 0:
            print(count)
        to_matlab['id'].append(int(count))
        to_matlab['event_id'].append(str(event.r0.event_id) + '03')
        to_matlab['label'].append(2.0)
        energy=event.mc.energy.to(unit.GeV)
        print(energy.value)
        to_matlab['mc_energy'].append(energy.value)
        hasdata = np.arange(4)

        for i in event.dl0.tels_with_data:
            hasdata[i - 1] = 1

        # Create arrays for event.
        to_matlab['HasData'].append(hasdata)
        tel_labels = np.zeros((no_tels, 1))
        datas = np.zeros((no_tels, 48, 48, 96))
        integrated = np.zeros((no_tels, 48, 48))
        timesarr = np.zeros((no_tels, 48, 48))
        fwhmarr = np.zeros((no_tels, 48, 48))
        rtarr = np.zeros((no_tels, 48, 48))
        ftarr = np.zeros((no_tels, 48, 48))
        meanarr = np.zeros((no_tels, 48, 48))
        rmsarr = np.zeros((no_tels, 48, 48))
        amparr = np.zeros((no_tels, 48, 48))
        cubed_list = []  # For plotting code

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

            if plot_wf == True:
                # Code to plot waveform parameters.
                plt.subplot(8, 1, 1)
                times = squared
                plt.imshow(times)
                plt.title('Charge')
                plt.axis('off')
                plt.subplot(8, 1, 2)
                times = cam_squaremaker(meanmat)
                plt.imshow(times)
                plt.title('Waveform Mean')
                plt.axis('off')
                plt.subplot(8, 1, 3)
                times = cam_squaremaker(ampmat)
                plt.imshow(times)
                plt.title('Pulse Amplitude')
                plt.axis('off')

                plt.subplot(8, 1, 4)
                times = ptimes
                plt.imshow(times[:, :, 0])
                plt.title('Pulse Time')
                plt.axis('off')

                plt.subplot(8, 1, 5)
                times = cam_squaremaker(rmsmat)
                plt.imshow(times)
                plt.title('Waveform RMS')
                plt.axis('off')

                plt.subplot(8, 1, 6)
                times = cam_squaremaker(np.asarray(fwhmmat))
                plt.imshow(times)
                plt.title('FWHM')
                plt.axis('off')

                plt.subplot(8, 1, 7)
                times = cam_squaremaker(np.asarray(rtmat))
                plt.imshow(times)
                plt.title('RT')
                plt.axis('off')

                plt.subplot(8, 1, 8)
                times = cam_squaremaker(np.asarray(ftmat))
                plt.imshow(times)
                plt.title('FT')
                plt.subplots_adjust(hspace=0.5)
                plt.axis('off')
                plt.show()

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

            if plot == True and count > event_plot and len(
                    event.r0.tels_with_data) > 3 and gammaflag == 2:
                cubed_list.append(cubed)
            else:
                continue

        for i in np.arange(4):
            if hasdata[i] == 0:
                tel_labels[i] = -1

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

        if plot == True and count > event_plot and len(
                event.r0.tels_with_data) > 3 and gammaflag == 2:
            plot_cubed(cubed_list)
            break

        count = count + 1

    pyhessio.close_file()

    # Make everything arrays in order to randomize.
    to_matlab['id'] = np.asarray(to_matlab['id'])
    to_matlab['event_id'] = np.asarray(to_matlab['event_id'])
    to_matlab['label'] = np.asarray(to_matlab['label'])
    to_matlab['mc_energy'] = np.asarray(to_matlab['mc_energy'])
    to_matlab['HasData'] = np.asarray(to_matlab['HasData'])
    to_matlab['tel_labels'] = np.asarray(to_matlab['tel_labels'])
    to_matlab['tel_integrated'] = np.asarray(to_matlab['tel_integrated'])
    to_matlab['peak_times'] = np.asarray(to_matlab['peak_times'])
    to_matlab['FWHM'] = np.asarray(to_matlab['FWHM'])
    to_matlab['RT'] = np.asarray(to_matlab['RT'])
    to_matlab['FT'] = np.asarray(to_matlab['FT'])
    to_matlab['waveform_mean'] = np.asarray(to_matlab['waveform_mean'])
    to_matlab['waveform_rms'] = np.asarray(to_matlab['waveform_rms'])
    to_matlab['waveform_amplitude'] = np.asarray(
        to_matlab['waveform_amplitude'])

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

    h5file = h5py.File(smallfile_name, "w")

    print('Writing')

    # HDF5 writer code
    lab_event = h5file.create_dataset(
        'event_label', (no_events,), dtype='i', compression="gzip")
    id_group = h5file.create_dataset(
        'id', (no_events,), dtype='i', compression="gzip")
    id_group2 = h5file.create_dataset(
        'event_id', (no_events,), dtype='i', compression="gzip")
    energy_group = h5file.create_dataset(
        'mc_energy', (no_events,), dtype='f', compression="gzip")

    squared_group = h5file.create_dataset(
        "squared_training", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    labels_group = h5file.create_dataset(
        "labels", (no_events, 4, 1), dtype='i', compression="gzip")
    times_group = h5file.create_dataset(
        "peak_times", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    fwhm_group = h5file.create_dataset(
        "FWHM", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    rt_group = h5file.create_dataset(
        "RT", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    ft_group = h5file.create_dataset(
        "FT", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    mean_group = h5file.create_dataset(
        "waveform_mean", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    rms_group = h5file.create_dataset(
        "waveform_rms", (no_events, 4, 48, 48), dtype='f', compression="gzip")
    amp_group = h5file.create_dataset(
        "waveform_amplitude", (no_events, 4, 48, 48), dtype='f', compression="gzip")

    for index in np.arange(0, no_events):
        index = int(index)
        lab_event[index] = np.int64(to_matlab['label'][index])
        id_group[index] = np.int64(to_matlab['id'][index])
        id_group2[index] = np.int64(to_matlab['event_id'][index])
        energy_group[index] = np.float64(to_matlab['mc_energy'][index])
        squared_group[index, :, :, :] = to_matlab['tel_integrated'][index]
        labels_group[index, :, :] = to_matlab['tel_labels'][index]
        times_group[index, :, :, :] = to_matlab['peak_times'][index]
        fwhm_group[index, :, :, :] = to_matlab['FWHM'][index]
        rt_group[index, :, :, :] = to_matlab['RT'][index]
        ft_group[index, :, :, :] = to_matlab['FT'][index]
        mean_group[index, :, :, :] = to_matlab['waveform_mean'][index]
        rms_group[index, :, :, :] = to_matlab['waveform_rms'][index]
        amp_group[index, :, :, :] = to_matlab['waveform_amplitude'][index]

    h5file.close()
endtime=time.time()
runtime=endtime-starttime
print('Time for 10 events to be written', runtime)
plt.show()
