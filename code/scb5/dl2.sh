#!/bin/bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=$HOME/student_behavior/hf_home
cd $HOME/student_behavior/scb_mosaic
rm -f SCB-public-mosaic.zip
$HOME/student_behavior/venv/bin/hf download wintonYF/SCB-public-mosaic SCB-public-mosaic.zip --repo-type dataset --local-dir .
echo MOSAIC_DL_DONE
ls -la
