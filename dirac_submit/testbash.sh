#!/bin/sh

VAR1=$1
VAR2=$2
VAR3=$3
echo $VAR1
echo $VAR2
echo $VAR3
VAR4=$VAR1$VAR2$VAR3
echo $VAR4
SIMTEL_FILE=( `find . -name "run1.txt"` )
echo $SIMTEL_FILE
