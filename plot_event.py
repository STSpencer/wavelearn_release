import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import h5py

 #Event to plot
file='/store/spencers/Data/pointrun3/265260186.hdf5'
inputdata = h5py.File(file, 'r')
trainarr = np.asarray(inputdata['peak_times'][:, :, :, :])
trainarr = trainarr[:, :, 8:40, 8:40]
chargearr = np.asarray(inputdata['squared_training'][:, :, :, :])
chargearr = chargearr[:, :, 8:40, 8:40]
rtarr = np.asarray(inputdata['RT'][:, :, :, :])
rtarr = rtarr[:, :, 8:40, 8:40]
ftarr = np.asarray(inputdata['FT'][:, :, :, :])
ftarr = ftarr[:, :, 8:40, 8:40]
fwhmarr = np.asarray(inputdata['FWHM'][:, :, :, :])
fwhmarr = fwhmarr[:, :, 8:40, 8:40]
amparr = np.asarray(inputdata['waveform_amplitude'][:, :, :, :])
amparr = amparr[:, :, 8:40, 8:40]
meanarr = np.asarray(inputdata['waveform_mean'][:, :, :, :])
meanarr = meanarr[:, :, 8:40, 8:40]
rmsarr = np.asarray(inputdata['waveform_rms'][:, :, :, :])
rmsarr = rmsarr[:, :, 8:40, 8:40]
labelsarr = np.asarray(inputdata['event_label'][:])
idarr = np.asarray(inputdata['id'][:])
energy=np.asarray(inputdata['mc_energy'][:].tolist())
inputdata.close()
print(np.where(energy==np.amax(energy)))
eventno=np.where(energy>40000)[0][0]
lendat=len(idarr)
trainarr = np.reshape(trainarr, (lendat, 4, 32, 32, 1))
chargearr = np.reshape(chargearr, (lendat, 4, 32, 32, 1))
rtarr = np.reshape(rtarr, (lendat, 4, 32, 32, 1))
ftarr = np.reshape(ftarr, (lendat, 4, 32, 32, 1))
fwhmarr = np.reshape(fwhmarr, (lendat, 4, 32, 32, 1))
amparr = np.reshape(amparr, (lendat, 4, 32, 32, 1))
meanarr = np.reshape(meanarr, (lendat, 4, 32, 32, 1))
rmsarr = np.reshape(rmsarr, (lendat, 4, 32, 32, 1))
fig,axes=plt.subplots(nrows=2,ncols=4,figsize=(8,4))
# Code to plot waveform parameters.
print(labelsarr[eventno],energy[eventno])
squared=chargearr[eventno,0,:,:,0]
axes[0,0].imshow(squared)
axes[0,0].set_title('Charge')
axes[0,0].axis('off')
ptimes=trainarr[eventno,0,:,:,0]
axes[0,1].imshow(ptimes)
axes[0,1].set_title('Peak Time')
axes[0,1].axis('off')
meanmat=meanarr[eventno,0,:,:,0]
axes[0,2].imshow(meanmat)
axes[0,2].set_title('Mean Amplitude')
axes[0,2].axis('off')
ampmat=amparr[eventno,0,:,:,0]
axes[0,3].imshow(ampmat)
axes[0,3].set_title('Peak Amplitude')
axes[0,3].axis('off')    
rmsmat=rmsarr[eventno,0,:,:,0]
axes[1,0].imshow(rmsmat)
axes[1,0].set_title('RMS')
axes[1,0].axis('off')
fwhmmat=fwhmarr[eventno,0,:,:,0]
axes[1,1].imshow(fwhmmat)
axes[1,1].set_title('FWHM')
axes[1,1].axis('off')    
rtmat=rtarr[eventno,0,:,:,0]
axes[1,2].imshow(rtmat)
axes[1,2].set_title('RT')
axes[1,2].axis('off')
ftmat=ftarr[eventno,0,:,:,0]
axes[1,3].imshow(ftmat)
axes[1,3].set_title('FT')
axes[1,3].axis('off')
plt.show()
