import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import glob
import h5py
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large'}
pylab.rcParams.update(params)

diffusefiles = glob.glob('/mnt/extraspace/exet4487/diffuserun1/*.hdf5')
diffusefiles=sorted(diffusefiles)

global diffusetruth

diffusetruth=[]

for file in diffusefiles[140:239]:
    inputdata = h5py.File(file, 'r')
    labelsarr = np.asarray(inputdata['event_label'][:])
    for value in labelsarr:
        diffusetruth.append(value)
    inputdata.close()

n_classes = 3

def rocplotter(axes,runname):
    pred=np.load('/users/exet4487/predictions/'+str(runname)+'_predictions.npy')
    roc_auc = dict()
    lw=2
    mr=[]
    mraw=[]
    for x in pred:
        mr.append(np.argmax(x))
        mraw.append(x)
    mraw=np.asarray(mraw)
    truth=diffusetruth
    no_ev=min([len(mr),len(truth)])
    mr=np.asarray(mr[:no_ev])
    mraw2=np.asarray(mraw[:no_ev])
    truth=np.asarray(truth)[:no_ev]
    gammas=np.where(truth==0)[0]
    protons=np.where(truth==1)[0]
    electrons=np.where(truth==2)[0]
    loc1=np.concatenate((gammas,protons))
    loc1=np.sort(loc1)
    t1=truth[loc1]
    mraw2=mraw2[loc1]
    mr2=mr[loc1]
    mr2=np.asarray(mr2)
    nonel=np.where(mr2!=2)
    nonel=np.sort(nonel)
    t1=t1[nonel]
    mr2=mr2[nonel][0]
    mraw2=1.0-mraw2[nonel][0]
    t1=np.squeeze(t1)
    #np.set_printoptions(threshold=np.inf)
    #mr2=label_binarize(mr2,classes=[0,1])
    print(t1,mr2)
    print(mraw2)
    print(np.shape(t1),np.shape(mraw2))
    fp1,tp1,_=roc_curve(t1,mraw2[:,0])
    auc1=auc(fp1,tp1)
    axes.plot(fp1,tp1,label='$\gamma$ v p, AUC=%.2f'%auc1,lw=lw)
    t2=label_binarize(truth,classes=[0,1,2])
    #mr3=label_binarize(mr,classes=[0,1,2])
    print(t2[:,1],mraw[:,1])
    fp2,tp2,_=roc_curve(t2[:,1],mraw[:,1])
    auc2=auc(fp2,tp2)
    print(auc2)
    axes.plot(fp2,tp2,label='p v ($\gamma$+e). AUC=%.2f'%auc2,lw=lw)
    print(auc2)
        
    truth=np.asarray(diffusetruth)[:no_ev]
    loc1=np.concatenate((gammas,electrons))
    loc1=np.sort(loc1)
    t1=truth[loc1]
    mraw2=np.asarray(mraw[:no_ev])[loc1]
    mr2=mr[loc1]
    mr2=np.asarray(mr2)
    nonel=np.where(mr2!=1)
    nonel=np.sort(nonel)
    t1=t1[nonel]
    mr2=mr2[nonel][0]
    mraw2=1.0-mraw2[nonel][0]
    t1=np.squeeze(t1)
    #np.set_printoptions(threshold=np.inf)
    #mr2=label_binarize(mr2,classes=[0,1])
    print('t1',t1,mr2)
    print(mraw2)
    print(np.shape(t1),np.shape(mraw2))
    t1=label_binarize(t1,classes=[0,2])
    fp1,tp1,_=roc_curve(t1,mraw2[:,0])
    auc1=auc(fp1,tp1)
    axes.plot(fp1,tp1,label='$\gamma$ v e, AUC=%.2f'%auc1,lw=lw)
    print('Diffuse')
    axes.legend(loc="lower right",fontsize='large')

def rp2(axes,runname):
    tpr=np.load('/users/exet4487/confmatdata/'+str(runname)+'_tp.npy',allow_pickle=True)
    fpr=np.load('/users/exet4487/confmatdata/'+str(runname)+'_fp.npy',allow_pickle=True)
    roc_auc = dict()
    lw=2
    print(fpr,tpr)
    for i in range(3):
        print(str(i))
        print(fpr.item().get(i))
        roc_auc[i] = auc(fpr.item().get(i), tpr.item().get(i))

    roc_auc["macro"] = auc(fpr.item().get("macro"), tpr.item().get("macro"))
    print(tpr,fpr,type(tpr),type(fpr))
    axes.plot(fpr.item().get("macro"), tpr.item().get("macro"),
             label='Average (AUC = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])                                                                                   
    classes=['$\gamma$','p','e']
    for i in range(n_classes):
        axes.plot(fpr.item().get(i), tpr.item().get(i), lw=lw,
                  label='{0} vs non-{0} (AUC = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    axes.legend(loc="lower right")

cols = range(0, 2)
rows = ['Method {}'.format(row) for row in ['A', 'B', 'C', 'D']]
xtitles = ['Diffuse Run (ES)','Diffuse Run (ES,Partitioned)']

from matplotlib.transforms import offset_copy

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
plt.setp(axes.flat, xlabel='FPR', ylabel='TPR')
pad = 5 # in points

for ax, col in zip(axes[0], cols):
    ax.annotate(xtitles[col], xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='x-large', ha='center', va='baseline')

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='x-large', ha='right', va='center')


rp2(axes[0,0],'diffuserun_timeonlyES')
rocplotter(axes[0,1],'diffuserun_timeonlyES')
rp2(axes[1,0],'diffuserun_allparamsES')
rocplotter(axes[1,1],'diffuserun_allparamsES')
rp2(axes[2,0],'diffuserun_shilonES')
rocplotter(axes[2,1],'diffuserun_shilonES')
rp2(axes[3,0],'diffuserun_chargetimeES')
rocplotter(axes[3,1],'diffuserun_chargetimeES')


fig.tight_layout()
# tight_layout doesn't take these labels into account. We'll need 
# to make some room. These numbers are are manually tweaked. 
# You could automatically calculate them, but it's a pain.
fig.subplots_adjust(left=0.15, top=0.95)
plt.savefig('/users/exet4487/Figures/esplot.png')
#plt.show()
