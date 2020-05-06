import numpy as np

#Right, what do we need for this dirac automation code

#Automated general submission script knowing taking lists of known files, and start and end events.
#Need function to take say a ls>file list of files in a given folder and turn that into a list of named files

def listmaker(listfile):
    return open(listfile).read().splitlines()

def ciphergen(no_files):
    gind=np.arange(no_files)
    pind=np.arange(no_files)
    eind=np.arange(no_files)
    np.random.shuffle(gind)
    np.random.shuffle(pind)
    np.random.shuffle(eind)
    np.save('gind2.npy',gind)
    np.save('pind2.npy',pind)
    np.save('eind2.npy',eind)
    return 0

def runnumber(runstr):
    if len(runstr)==15:
        runno=int(runstr[3:5])
    elif len(runstr)==14:
        runno=int(runstr[3])
    elif len(runstr)==16:
        runno=int(runstr[3:6])
    else:
        print(runstr[-8])
        print(len(runstr))
        print(runstr)
        raise KeyboardInterrupt
    return runno

#Need then to create flists
no_gammafiles=4
no_protonfiles=4
no_electronfiles=10

gfilelist=['gamma_list_'+str(i)+'.txt' for i in np.arange(no_gammafiles)]
gammalist=[listmaker(gfile) for gfile in gfilelist]
gammaref=[]

for i in np.arange(len(gammalist)):
    for j in gammalist[i]:
        if j[-8]!='h':
            rno=runnumber(j)
            gammaref.append(str(i)+'_'+str(rno))

print('gamma',gammaref)

pfilelist=['proton_list_'+str(i)+'.txt' for i in np.arange(no_protonfiles)]
protonlist=[listmaker(pfile) for pfile in pfilelist]
protonref=[]

for i in np.arange(len(protonlist)):
    for j in protonlist[i]:
        if j[-8]!='h':
            rno=runnumber(j)
            protonref.append(str(i)+'_'+str(rno))

print('proton',protonref)

efilelist=['electron_list_'+str(i)+'.txt' for i in np.arange(no_electronfiles)]
electronlist=[listmaker(efile) for efile in efilelist]
electronref=[]

for i in np.arange(len(electronlist)):
    for j in electronlist[i]:
        if j[-8]!='h':
            rno=runnumber(j)
            electronref.append(str(i)+'_'+str(rno))

print('electron',electronref)
no_events=[len(gammaref),len(protonref),len(electronref)]
minno=min(no_events)

gammaref=np.asarray(gammaref[-minno:])
protonref=np.asarray(protonref[-minno:])
electronref=np.asarray(electronref[-minno:])
print(gammaref,protonref,electronref)
print(no_events)

gind=np.load('gind.npy')
pind=np.load('pind.npy')
eind=np.load('eind.npy')
gammaref=gammaref[gind]
protonref=protonref[pind]
electronref=electronref[eind]

def pathconstructor(classlabel,refstr):
    
    if classlabel==0:
        classstr='gamma'
    elif classlabel==1:
        classstr='proton'
    elif classlabel==2:
        classstr='electron'
    else:
        print('Invalid class')
        raise KeyboardInterrupt
    
    breakloc=refstr.index('_')
    foldindex=str(int(refstr[:breakloc])+1)
    fileno=refstr[breakloc+1:]
    path="LFN:/vo.cta.in2p3.fr/user/t/tarmstrong/sam_sims/simtel_data/"+classstr+foldindex+"/run"+fileno+".simtel.gz"
    return fileno,foldindex,path

failed=0
for k in np.arange(minno):
        gammano,gammaind,gammapath=pathconstructor(0,gammaref[k])
        protonno,protonind,protonpath=pathconstructor(1,protonref[k])
        electronno,electronind,electronpath=pathconstructor(2,electronref[k])
        if gammano==protonno or gammano==electronno:
            failed=failed+1
            continue
        else:
            runstr=gammaind+protonind+electronind+gammano+protonno+electronno
            print(runstr)
            print(gammapath)
            print(protonpath)
            print(electronpath)
        #Have if statement to check to runnos the same
        #Go through all three classes' randomized lists

print(failed)

# Have random cipher mixing up the three lists saved to disk.
# Split up identifier into folder and file number
# Have if statement to check file numbers not equal
# Pass to submission script


#proton_locs_0=listfile(proton_list_0)
#electron_locs_0=listfile(electron_list_0)

def autosubmit(gamma_locs,proton_locs,electron_locs,start,end):
    #Need to implement moving fences here so that no two files have the same runno
    for i in np.linspace(start,end):
        gamma_no=gamma_locs[i][1] #Need some way of finding number associated with particular file, may vary, may need separate fn.
        proton_no=proton_locs[i][1]
        electron_no=electron_locs[i][1]
    files_to_submit=0
    return files_to_submit

#Second function to handle single submissions that fail, preferably keep some sort of automated log
def singlesubmit(gammafile,protonfile,electronfile):
    file_to_submit=0
    return files_to_submit


#Need some way of labelling proton2, proton3 etc in filenames.
