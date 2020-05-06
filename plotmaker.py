import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large'}

pylab.rcParams.update(params)
n_classes = 3

def rocplotter(axes,runname):
    tpr=np.load('/home/spencers/confmatdata/'+str(runname)+'_tp.npy',allow_pickle=True)
    fpr=np.load('/home/spencers/confmatdata/'+str(runname)+'_fp.npy',allow_pickle=True)
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
xtitles = ['Point Source Run','Diffuse Run']

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


rocplotter(axes[0,0],'pointrun_timeonly')
rocplotter(axes[0,1],'diffuserun_timeonly')
rocplotter(axes[1,0],'pointrun_allparams')
rocplotter(axes[1,1],'diffuserun_allparams')
rocplotter(axes[2,0],'pointrun_shilon')
rocplotter(axes[2,1],'diffuserun_shilon')
rocplotter(axes[3,0],'pointrun_chargetime')
rocplotter(axes[3,1],'diffuserun_chargetime')


fig.tight_layout()
# tight_layout doesn't take these labels into account. We'll need 
# to make some room. These numbers are are manually tweaked. 
# You could automatically calculate them, but it's a pain.
fig.subplots_adjust(left=0.15, top=0.95)
plt.savefig('/home/spencers/Figures/final_newlabels.png')
#plt.show()
