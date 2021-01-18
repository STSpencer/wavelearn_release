import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import numpy as np

filelist = sorted(glob.glob('/mnt/extraspace/exet4487/diffuserun1/*.hdf5'))

filelist=filelist[:1]

def img_clean(image,highthresh,lowthresh):
    flatim=image.flatten()
    flatim=0.01525723*flatim-6.2041985 #Correction to p.e. for prod3b chec with ctapipe 0.6.1
    mask=[]
    
    overeight=np.array(np.where(flatim>highthresh))
    oversix=np.array(np.where(flatim>lowthresh))
    
    for i in overeight:
        mask.append(i)
    for j in oversix:
        if j-1 in overeight:
            mask.append(j)
        if j+1 in overeight:
            mask.append(j)
    mask=np.asarray(mask)[0]
    mask=np.unique(mask)
    for i in np.arange(len(flatim)):
        if i in mask:
            continue
        else:
            flatim[i]=0.0
    return flatim

def hillas(image,pix_x,pix_y):
    HILLAS_ATOL = np.finfo(np.float64).eps
    image = np.asanyarray(image, dtype=np.float64)
    image = np.ma.filled(image, 0)
    msg = "Image and pixel shape do not match"
    pix_x=pix_x[512:1536]
    pix_y=pix_y[512:1536]
    assert pix_x.shape == pix_y.shape == image.shape, msg

    size = np.sum(image)

    if size == 0.0:
        raise HillasParameterizationError("size=0, cannot calculate HillasParameters")

    # calculate the cog as the mean of the coordinates weighted with the image
    cog_x = np.average(pix_x,weights=image)
    cog_y = np.average(pix_y,weights=image)

    # polar coordinates of the cog
    cog_r = np.linalg.norm([cog_x, cog_y])
    cog_phi = np.arctan2(cog_y, cog_x)

    # do the PCA for the hillas parameters
    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    # The ddof=0 makes this comparable to the other methods,
    # but ddof=1 should be more correct, mostly affects small showers
    # on a percent level
    cov = np.cov(delta_x, delta_y, aweights=image, ddof=0)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # round eig_vals to get rid of nans when eig val is something like -8.47032947e-22
    near_zero = np.isclose(eig_vals, 0, atol=HILLAS_ATOL)
    eig_vals[near_zero] = 0

    # width and length are eigen values of the PCA
    width, length = np.sqrt(eig_vals)

    return width,length
    
def generate_training_data(filelist):
    """ Generates training/test sequences on demand
    """

    allcharges=[]
    alllabels=[]
    pix_x = np.load('pix_x.npy')
    pix_y = np.load('pix_y.npy')
    for file in filelist:
        inputdata = h5py.File(file, 'r')
        print(np.shape(inputdata['event_label']))
        for j in np.arange(np.shape(inputdata['event_label'])[0]):
            chargearr = inputdata['squared_training'][j, 0,8:40,8:40]
            clean=img_clean(chargearr,8.0,6.0)
            size=np.sum(clean)
            if size>0:
                hillasw,hillasl=hillas(clean,pix_x,pix_y)
                print('hillas',hillasw,hillasl)
            '''
            fig=plt.figure()
            plt.imshow(clean)
            plt.savefig('/users/exet4487/cleanfigs/'+str(j)+'.png')
            '''
            labelsarr = inputdata['event_label'][j]
            allcharges.append(chargearr)
            alllabels.append(labelsarr)
    allcharges=np.asarray(allcharges)
    alllabels=np.asarray(alllabels)
    return allcharges, alllabels

x,y=generate_training_data(filelist)
print(np.shape(x),np.shape(y))
print(x[0])
print(y)
