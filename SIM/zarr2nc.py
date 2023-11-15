#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert .zarr files to .nc (for CEDA)
@author: Noam Vogt-Vincent
"""

import os
import xarray as xr
from tqdm import tqdm
from glob import glob

# Get a list of all .zarr files in directory and create new file handle
fh_i_list = sorted(glob(os.getcwd() + '/*.zarr'))
fh_o_list = [fh.replace('.zarr', '.nc') for fh in fh_i_list]

# Loop through files and export to netcdf
for fh_i, fh_o in tqdm(zip(fh_i_list, fh_o_list), total=len(fh_i_list)):
    file = xr.open_zarr(fh_i)
    file.to_netcdf(fh_o, encoding={var: {'zlib': True, 'complevel': 5} for var in file.variables})