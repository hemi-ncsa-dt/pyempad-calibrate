# EMPAD Calibrate

> [!NOTE]
> Cython reimplementation of MATLAB scripts for calibrating EMPAD data. See the original repository at 
> https://github.com/paradimdata/pyempadcalibratescript

This package is a Python version of exiting a MATLAB code for Electron Microscope Pixel Array Detector (EMPAD)
Calibration. The original MATLAB code was based on [Philipp et al., [2022]](https://doi.org/10.1017/S1431927622000174).

### Usage

```bash

pyempad-calibrate \
  --calib_path <directory_with_calibration_data> \
  background_scan_x256_y256.raw \
  data_scan_x256_y256.raw

```
