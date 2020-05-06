#!/bin/sh
export MINICONDA=/cvmfs/cta.in2p3.fr/software/miniconda
source $MINICONDA/setupConda.sh
source $MINICONDA/bin/activate ctapipe_v0.6.1 
#source $MINICONDA/bin/activate ctapipe_v0.5.3

VAR1=$1
VAR2=$2
VAR3=$3
VAR4=$VAR1$VAR2$VAR3
echo $VAR1
echo $VAR2
echo $VAR3
echo $VAR4

GAMMA_FILE=( `find . -name "run$VAR1\_merged.simtel.gz"` )
HADRON_FILE=( `find . -name "run$VAR2\_merged.simtel.gz"` )
ELECTRON_FILE=( `find . -name "run$VAR3\_merged.simtel.gz"` )

echo $GAMMA_FILE
echo $HADRON_FILE
echo $ELECTRON_FILE

echo simtel_writer_dirac.py $GAMMA_FILE $HADRON_FILE $ELECTRON_FILE $VAR4

python simtel_writer_dirac.py $GAMMA_FILE $HADRON_FILE $ELECTRON_FILE $VAR4 > simtel_writer.log 2>&1
