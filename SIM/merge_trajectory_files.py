#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge trajectory files
@author: Noam
"""

import xarray as xr
import numpy as np
import os
from glob import glob
from sys import argv
from tqdm import tqdm

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['traj_in'] = dirs['root'] + 'TRAJ/'
dirs['traj_out'] = dirs['root'] + 'TRAJ/POSTPROC/'

# PATTERN
pattern = 'WINDS_coralsim_' #'WINDS_coralsim_2020_1_3_1_2.nc
years = [int(argv[1])]

##############################################################################
# LOOP THROUGH FILES AND MERGE                                               #
##############################################################################

# fh_list = sorted(glob(dirs['traj_in'] + pattern + '*'))
# years = np.unique([int(item.split('/')[-1].split('_')[-5]) for item in fh_list])

for year in years:
    # Get list of all files from year
    fh_list_year = sorted(glob(dirs['traj_in'] + pattern + str(year) + '_*'))
    months = np.unique([int(item.split('/')[-1].split('_')[-4]) for item in fh_list_year])

    pbar = tqdm(total=len(fh_list_year))

    for month in months:
        # Get list of all files from month
        fh_list_month = sorted(glob(dirs['traj_in'] + pattern + str(year) + '_' + str(month) + '_*'))
        days = np.unique([int(item.split('/')[-1].split('_')[-3]) for item in fh_list_month])

        for day in days:
            # Get list of all files from day
            fh_list_day = sorted(glob(dirs['traj_in'] + pattern + str(year) + '_' + str(month) + '_' + str(day) + '_*'))
            part_list = np.unique([int(item.split('/')[-1].split('_')[-1].split('.')[0]) for item in fh_list_day])
            assert len(part_list) == 1 # Check there is no variable number of partitions
            assert len(fh_list_day) == part_list[0] # Check all files are present

            for fhi, fh in enumerate(fh_list_day):
                file = xr.open_dataset(fh, mask_and_scale=False)

                if fhi == 0:
                    var_dict = {}
                    attr_dict = {}
                    e_num = int(file.attrs['e_num'])

                for v_name in ['e_num', 'idx0', 'lon0', 'lat0']:
                    if fhi == 0:
                        var_dict[v_name] = file[v_name].values
                    else:
                        var_dict[v_name] = np.concatenate((var_dict[v_name], file[v_name].values))

                for attr_name in ['parcels_version', 'timestep_seconds',
                                  'min_competency_seconds', 'max_lifespan_seconds',
                                  'larvae_per_cell', 'interp_method', 'e_num',
                                  'release_year', 'release_month', 'release_day']:
                    if fhi == 0:
                        attr_dict[attr_name] = file.attrs[attr_name]
                    else:
                        assert attr_dict[attr_name] == file.attrs[attr_name]

                if fhi == 0:
                    attr_dict['total_larvae_released'] = file.attrs['total_larvae_released']
                else:
                    attr_dict['total_larvae_released'] += file.attrs['total_larvae_released']

                for var_type in ['i', 'ts', 'dt']:
                    for e_idx in range(e_num):
                        v_name = var_type + str(e_idx)

                        if fhi == 0:
                            var_dict[v_name] = file[v_name].values
                        else:
                            var_dict[v_name] = np.concatenate((var_dict[v_name], file[v_name].values))

                pbar.update(1)

            for v_name in var_dict.keys():
                var_dict[v_name] = ('traj', var_dict[v_name].flatten())

            new_file = xr.Dataset(data_vars=var_dict,
                                  coords={'traj': np.arange(len(var_dict['e_num'][1]))},
                                  attrs=attr_dict)

            new_fh = pattern + str(year) + '_' + str(month) + '_' + str(day) + '.zarr'

            new_file.to_zarr(store=dirs['traj_out'] + new_fh, mode='w',
                             consolidated=True)