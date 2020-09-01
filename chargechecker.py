import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob

onlyfiles = sorted(glob.glob('/store/spencers/Data/pointrun3/*.hdf5'))
chargesums=[]
cutsums=[]
chargesums2=[]
cutsums2=[]
count=0
count2=0
bins=np.linspace(10**5.5,10**8)
for file in onlyfiles:
    count=count+1
    print(count,'/',len(onlyfiles))
    inputdata = h5py.File(file, 'r')
    chargearr = np.asarray(inputdata['squared_training'][:, :, :, :])
    evlabels=np.asarray(inputdata['event_label'][:])
    for i in np.arange(np.shape(chargearr)[0]):
        '''
        print(evlabels[i])
        plt.imshow(chargearr[i,0,:,:])
        #Center region crop
        plt.hlines(8,8,40,colors='r')
        plt.hlines(40,8,40,colors='r')
        plt.vlines(8,8,40,colors='r')
        plt.vlines(40,8,40,colors='r')
        plt.xlabel('x (Pixels)',size='large')
        plt.ylabel('y (Pixels)',size='large')'''
        '''
        plt.hlines(8,0,8,colors='b')
        plt.hlines(8,40,47,colors='b')
        plt.hlines(40,0,8,colors='b')
        plt.hlines(40,40,47,colors='b')
        plt.vlines(8,0,8,colors='b')
        plt.vlines(8,40,47,colors='b')
        plt.vlines(40,0,8,colors='b')
        plt.vlines(40,40,47,colors='b')'''
        plt.show()
        chargesums.append(np.sum(chargearr[i,:,:,:]))
        cutsums.append(np.sum(chargearr[i,:,8:40,8:40]))
    inputdata.close()

fig, (ax1, ax2) = plt.subplots(1, 2)
chargesums=np.asarray(chargesums)
cutsums=np.asarray(cutsums)
ax1.hist(chargesums,bins=bins,label='48x48',alpha=0.2)
ax1.hist(cutsums,bins=bins,label='32x32',alpha=0.2)
ax1.set_xlabel('Total Charge (Relative Units)',size='large')
ax1.set_ylabel('Number of Events (Counts)',size='large')
ax1.set_title('Point Source Run',size='large')
ax1.loglog()
ax1.legend(fontsize='large')
ax1.set_xlim(630957,1e8)
onlyfiles = sorted(glob.glob('/store/spencers/Data/diffuserun1/*.hdf5'))

for file in onlyfiles:
    count2=count2+1
    print(count2,'/',len(onlyfiles))
    inputdata = h5py.File(file, 'r')
    chargearr = np.asarray(inputdata['squared_training'][:, :, :, :])
    for i in np.arange(np.shape(chargearr)[0]):
        chargesums2.append(np.sum(chargearr[i,:,:,:]))
        cutsums2.append(np.sum(chargearr[i,:,8:40,8:40]))
    inputdata.close()

chargesums2=np.asarray(chargesums2)
cutsums2=np.asarray(cutsums2)
ax2.hist(chargesums2,bins=bins,label='48x48',alpha=0.2)
ax2.hist(cutsums2,bins=bins,label='32x32',alpha=0.2)
ax2.set_xlabel('Total Charge (Relative Units)',size='large')
ax2.set_ylabel('Number of Events (Counts)',size='large')
ax2.set_title('Diffuse Run',size='large')
ax2.loglog()
ax2.set_xlim(630957,1e8)
ax2.legend(fontsize='large')
plt.tight_layout()
plt.savefig('/home/spencers/Figures/chargehistlog10.png',figsize=(8.27, 14), dpi=300)
plt.show()

