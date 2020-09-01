import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import h5py

import matplotlib.ticker as mtick
def div_5(x, *args):
    """
    The function that will you be applied to your y-axis ticks.
    """
    x = float(x)/5.0
    return "{:.1f}".format(x)

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
fig,axes=plt.subplots(nrows=2,ncols=4,figsize=(16,8))
# Code to plot waveform parameters.
print(labelsarr[eventno],energy[eventno])
squared=chargearr[eventno,0,:,:,0]
im=axes[0,0].imshow(squared)
axes[0,0].set_title('Charge')
axes[0,0].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[0,0].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
#axes[0,0].axis('off')
cbar=fig.colorbar(im,ax=axes[0,0],fraction=0.046, pad=0.04)
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
ptimes=trainarr[eventno,0,:,:,0]
im=axes[0,1].imshow(ptimes)
axes[0,1].set_title('Peak Time')
axes[0,1].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[0,1].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
#axes[0,1].axis('off')
cbar=fig.colorbar(im,ax=axes[0,1],fraction=0.046, pad=0.04)
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
meanmat=meanarr[eventno,0,:,:,0]
im=axes[0,2].imshow(meanmat)
axes[0,2].set_title('Mean Amplitude')
axes[0,2].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[0,2].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
#axes[0,2].axis('off')
cbar=fig.colorbar(im,ax=axes[0,2],fraction=0.046, pad=0.04)
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
ampmat=amparr[eventno,0,:,:,0]
im=axes[0,3].imshow(ampmat)
axes[0,3].set_title('Peak Amplitude')
#axes[0,3].axis('off')
axes[0,3].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[0,3].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
cbar=fig.colorbar(im,ax=axes[0,3],fraction=0.046, pad=0.04)    
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
rmsmat=rmsarr[eventno,0,:,:,0]
im=axes[1,0].imshow(rmsmat)
axes[1,0].set_title('RMS')
#axes[1,0].axis('off')
axes[1,0].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[1,0].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
cbar=fig.colorbar(im,ax=axes[1,0],fraction=0.046, pad=0.04)
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
fwhmmat=fwhmarr[eventno,0,:,:,0]
im=axes[1,1].imshow(fwhmmat)
axes[1,1].set_title('FWHM')
#axes[1,1].axis('off')
axes[1,1].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[1,1].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
cbar=fig.colorbar(im,ax=axes[1,1],fraction=0.046, pad=0.04)
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
rtmat=rtarr[eventno,0,:,:,0]
im=axes[1,2].imshow(rtmat)
axes[1,2].set_title('RT')
#axes[1,2].axis('off')
axes[1,2].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[1,2].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
cbar=fig.colorbar(im,ax=axes[1,2],fraction=0.046, pad=0.04)
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
ftmat=ftarr[eventno,0,:,:,0]
im=axes[1,3].imshow(ftmat)
axes[1,3].set_title('FT')
cbar=fig.colorbar(im,ax=axes[1,3],fraction=0.046, pad=0.04)
#axes[1,3].axis('off')
#cbar.ax.set_ylabel('Value (Relative Units)', rotation=270)
axes[1,3].yaxis.set_major_formatter(mtick.FuncFormatter(div_5))
axes[1,3].xaxis.set_major_formatter(mtick.FuncFormatter(div_5))
for i, row in enumerate(axes):
    for j, cell in enumerate(row):
        if i == len(axes) - 1:
            cell.set_xlabel("Angular Size ($^\circ$)".format(j + 1),size='large')
        if j == 0:
            cell.set_ylabel("Angular Size ($^\circ$)".format(i + 1),size='large')

plt.tight_layout(0.5)
#plt.subplots_adjust(hspace=0.35,
 #                   wspace=0.6)

plt.savefig('56tgamma2.png')
