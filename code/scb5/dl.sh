#!/bin/bash
cd ~/student_behavior/scb_mosaic
curl -fL --retry 5 --retry-delay 5 -C - -o SCB-public-mosaic.zip https://huggingface.co/datasets/wintonYF/SCB-public-mosaic/resolve/main/SCB-public-mosaic.zip
echo MOSAIC_DL_DONE
ls -la
