#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export trajectory data from WINDS coralsim runs
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from sys import argv
from datetime import datetime

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
bio_code = argv[1]

# OPTIONS
opt = {}
opt['flux_src_grp'] = True
opt['drift_time_grp'] = True
opt['src_str_cell'] = True
opt['snk_str_cell'] = True
opt['ret_str_cell'] = True
opt['src_ent_cell'] = True
opt['snk_ent_cell'] = True
opt['flux_src_cell'] = True

opt['src_cru_cell'] = True
opt['snk_cru_cell'] = True
opt['src_tst_cell'] = True
opt['snk_tst_cell'] = True

opt['convert_daily'] = False # Convert flux_src_cell units to day-1

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['input'] = dirs['root'] + 'MATRICES/STEP_1/'
dirs['output'] = dirs['root'] + 'MATRICES/'

# FILES
name_list = ['flux_src_grp', 'drift_time_grp', 'src_str_cell', 'snk_str_cell',
             'ret_str_cell', 'src_ent_cell', 'snk_ent_cell', 'flux_src_cell',
             'src_cru_cell', 'snk_cru_cell', 'src_tst_cell', 'snk_tst_cell']

fh_list_in = {name: sorted(glob(dirs['input'] + 'WINDS_' + name + '_submatrix_' + bio_code + '_*.nc')) for name in name_list if opt[name]}
fh_list_out = {name: dirs['output'] + 'WINDS_' + name + '_' + bio_code + '.nc' for name in name_list if opt[name]}

##############################################################################
# COMBINE FILES                                                              #
##############################################################################

# List of concatenative variables (i.e. each file contains an annual time-series)
for concat_var in ['flux_src_grp', 'src_str_cell', 'snk_str_cell', 'ret_str_cell',
                   'src_ent_cell', 'snk_ent_cell', 'src_cru_cell', 'snk_cru_cell',
                   'src_tst_cell', 'snk_tst_cell']:
    if opt[concat_var]:
        # Check-variables
        year_list = []
        day_count = 0

        # Matrix list
        submatrix_list = []

        # Get file names
        fh_in = fh_list_in[concat_var]

        for fhi, fh in enumerate(fh_in):
            with xr.open_dataset(fh) as file:
                # Add year and day count to list
                year_list.append(int(fh.split('_')[-1].split('.')[0]))

                if 'time' in file.coords:
                    day_count += len(file.coords['time'])
                    daily = True
                elif 'month' in file.coords:
                    daily = False
                else:
                    raise ValueError('Time axis not understood.')

                # Check that attributes are the same
                if fhi == 0:
                    attr_dict = file.attrs
                else:
                    for attr in ['a', 'b', 'configuration', 'e_num', 'interp_method',
                                 'larvae_per_cell', 'max_lifespan_seconds',
                                 'min_competency_seconds', 'parcels_version', 'tc',
                                 'timestep_seconds', 'λ', 'μs', 'ν', 'σ']:
                        assert attr_dict[attr] == file.attrs[attr]

                    attr_dict['total_larvae_released'] += file.attrs['total_larvae_released']

                # Check number of days are correct
                try:
                    if daily:
                        assert len(file.coords['time']) == 365 if year_list[-1]%4 else 366
                        expected_release = attr_dict['larvae_per_cell']*8088*len(file.coords['time'])
                        assert file.attrs['total_larvae_released'] == expected_release
                    else:
                        assert len(file.coords['month']) == 12
                        n_day_year = 365 if year_list[-1]%4 else 366
                        expected_release = attr_dict['larvae_per_cell']*8088*n_day_year
                        assert file.attrs['total_larvae_released'] == expected_release
                except:
                    print('Incorrect number of particle releases in file ' + fh)
                    print('Expected number of particles: ' + str(expected_release))
                    print('Actual number of particles: ' + str(file.attrs['total_larvae_released']))
                    difference = expected_release - file.attrs['total_larvae_released']
                    print('Difference: ' + str(difference) + ' (' + str(difference/(attr_dict['larvae_per_cell']*8088)) + ' releases)')

                # If monthly, change the time axis
                if not daily:
                    file = file.assign_coords({'month': pd.date_range(start=datetime(year=year_list[-1], month=1, day=1),
                                                                      end=datetime(year=year_list[-1], month=12, day=31),
                                                                      freq='M')})

                # Add to concatenation list
                submatrix_list.append(file)

        # Concatenate and carry out checks
        matrix = xr.concat(submatrix_list, dim='time')
        matrix.attrs['total_larvae_released'] = attr_dict['total_larvae_released']

        year_list = np.sort(year_list)
        assert (np.unique(np.sort(year_list), return_counts=True)[1] == 1).all()
        assert (np.gradient(np.sort(year_list)) == 1).all()

        assert matrix.attrs == attr_dict
        assert day_count == len(matrix.coords['time'])
        assert matrix.attrs['total_larvae_released'] == attr_dict['larvae_per_cell']*8088*len(matrix.coords['time'])
        matrix.attrs['start_year'] = year_list[0]
        matrix.attrs['end_year'] = year_list[-1]

        # Export
        matrix.to_netcdf(fh_list_out[concat_var], encoding={var: {'zlib': True, 'complevel': 5} for var in matrix.variables})


# List of merging variables (i.e. each file contains an annual time-series)
for concat_var in ['drift_time_grp', 'flux_src_cell']:
    if opt[concat_var]:
        # Check-variables
        year_list = []
        day_count = 0

        # Matrix list
        submatrix_list = []

        # Get file names
        fh_in = fh_list_in[concat_var]

        for fhi, fh in enumerate(fh_in):
            with xr.open_dataset(fh) as file:
                # Add to year list
                year_list.append(int(fh.split('_')[-1].split('.')[0]))

                if 'drift_time' not in concat_var:
                    normal_mode = True
                else:
                    normal_mode = False

                if normal_mode:
                    if len(file.time) == 12:
                        monthly = True
                        file = file.assign_coords({'time': np.arange(1, 13)}).rename({'time': 'month'})
                    else:
                        monthly = False
                else:
                    monthly = False

                # Check that attributes are the same
                if fhi == 0:
                    attr_dict = file.attrs
                    if normal_mode and not monthly:
                        day_count = len(file.time)
                else:
                    for attr in ['a', 'b', 'configuration', 'e_num', 'interp_method',
                                 'larvae_per_cell', 'max_lifespan_seconds',
                                 'min_competency_seconds', 'parcels_version', 'tc',
                                 'timestep_seconds', 'λ', 'μs', 'ν', 'σ']:
                        assert attr_dict[attr] == file.attrs[attr]

                    if normal_mode and not monthly:
                        assert len(file.time) == day_count
                        # if not file.time.equals(matrix.time):
                        #     print()
                        assert file.time.equals(matrix.time)
                    elif monthly:
                        assert len(file.month) == 12

                    attr_dict['total_larvae_released'] += file.attrs['total_larvae_released']

                try:
                    days_in_year = 365 if year_list[-1]%4 else 366
                    expected_release = attr_dict['larvae_per_cell']*8088*days_in_year
                    assert file.attrs['total_larvae_released'] == expected_release
                except:
                    print('Incorrect number of particle releases in file ' + fh)
                    print('Expected number of particles: ' + str(expected_release))
                    print('Actual number of particles: ' + str(file.attrs['total_larvae_released']))
                    difference = expected_release - file.attrs['total_larvae_released']
                    print('Difference: ' + str(difference) + ' (' + str(difference/(attr_dict['larvae_per_cell']*8088)) + ' releases)')

                # Add to merge list
                if fhi == 0:
                    matrix = file
                else:
                    matrix.ns.data += file.ns.values

        year_list = np.sort(year_list)
        assert (np.unique(np.sort(year_list), return_counts=True)[1] == 1).all()
        assert (np.gradient(np.sort(year_list)) == 1).all()

        matrix.attrs = attr_dict
        matrix.attrs['total_larvae_released'] = attr_dict['total_larvae_released']
        matrix.attrs['start_year'] = year_list[0]
        matrix.attrs['end_year'] = year_list[-1]

        # In the case of the climatological flux cell matrix, normalise by
        # the number of days per month across the study period
        if concat_var == 'flux_src_cell':
            data_ymin = int(matrix.start_year)
            data_ymax = int(matrix.end_year)
            for month in range(1, 13):
                # Find number of days across study period
                if opt['convert_daily']:
                    t_axis = pd.date_range(start=datetime(year=data_ymin, month=1, day=1),
                                           end=datetime(year=data_ymax, month=12, day=31),
                                           freq='1D')
                    n_day_month = float(len(t_axis[t_axis.month == month]))
                else:
                    n_day_month = 1

                matrix.ns.loc[:, :, month] = matrix.ns.loc[:, :, month]/n_day_month

        # Export
        matrix.to_netcdf(fh_list_out[concat_var], encoding={var: {'zlib': True, 'complevel': 5} for var in matrix.variables})
