# wavelearn_release

Minimal code and trained models to recreate the results in the paper 'Deep learning with photosensor timing information as a background rejection method for the Cherenkov Telescope Array'.

**This code is not recommended for general CTA use, and is provided for legacy purposes only. For further machine learning research we recommend either CTLearn https://github.com/ctlearn-project/ctlearn or Gammalearn https://gitlab.lapp.in2p3.fr/GammaLearn/GammaLearn. This code is provided as is.**

The folder dirac_submit contains scripts to submit the .simtel.gz to hdf5 converter code simtel_writer_dirac.py to CTA-Dirac. Note that there are numerous complications involved with merging together proton,gamma and electron events on the grid, namely that there wind up being multiple 'runxx.simtel.gz' files with differing particles but the same name. As such, we use a lists of existing grid simtel files (i.e. proton_list_*.txt' to log existing files (as runs to generate simtel files can easily fail) and a cipher file (cipher.npy) to merge them together. simtel_writer.py is an equivalent script to run locally.

paramlstm.py is the main CRNN training script where the network architecture is defined. Note that numerous aspects are hardcoded, such as compatibility only with CHEC prod3b simulations. net_utils.py contains plotting and data generation functions needed for paramlstm.py. Separation of files into training/testing/validation data is performed by a cut on a filelist that must be the same in both paramlstm.py and net_utils.py. The different methods in the paper are implemented by changing the elements of the array ta2 and by changing the network input size in paramlstm.py as required. 

The folder Models contains pre-trained hdf5 Models for the methods presented in the paper.

plotmaker.py, abelardo.py, chargechecker.py and acclossplotter.py are helper scripts to recreate the plots in the paper.
