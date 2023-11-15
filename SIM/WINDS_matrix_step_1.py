#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export trajectory data from WINDS coralsim runs
@author: Noam Vogt-Vincent
"""

import os
import json
import xarray as xr
import numpy as np
from datetime import timedelta
from SECoW import Experiment
from sys import argv

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# PARAMETERS
bio_code = argv[1]
year = argv[2]
RK4_its = int(argv[3])

# OPTIONS
flux_src_grp = True
drift_time_grp = True
src_str_cell = True
snk_str_cell = True
ret_str_cell = True
src_ent_cell = True
snk_ent_cell = True
flux_src_cell = True

src_cru_cell = True
snk_cru_cell = True
src_tst_cell = True
snk_tst_cell = True

opt_name_list = {'flux_src_grp': flux_src_grp, 'drift_time_grp': drift_time_grp,
                 'src_str_cell': src_str_cell, 'snk_str_cell': snk_str_cell,
                 'ret_str_cell': ret_str_cell, 'src_ent_cell': src_ent_cell,
                 'snk_ent_cell': snk_ent_cell, 'flux_src_cell': flux_src_cell,
                 'src_cru_cell': src_cru_cell, 'snk_cru_cell': snk_cru_cell,
                 'src_tst_cell': src_tst_cell, 'snk_tst_cell': snk_tst_cell}

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/WINDS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/WINDS/'
dirs['ref'] = dirs['root'] + 'REFERENCE/'
dirs['output'] = dirs['root'] + 'MATRICES/'

# FILES
fh = {}
fh['TST'] = dirs['ref'] + 'TST/SECoW_TST.nc'
fh['CRU'] = dirs['ref'] + 'CRU/SECoW_CRU.nc'

##############################################################################
# FORMAT OUTPUT                                                              #
##############################################################################

# Get parameters from file
with open(dirs['ref'] + 'bio_parameters.json') as param_file:
    bio_data = json.load(param_file)[bio_code][0]

for key in ['a', 'b', 'μs', 'λ']:
    if bio_data[key] != 0:
        bio_data[key] = 1/timedelta(days=1/bio_data[key]).total_seconds()

bio_data['tc'] = timedelta(days=bio_data['tc']).total_seconds()

sey = Experiment('WINDS_' + bio_code)
sey.config(dirs, preset='WINDS', releases_per_month=1)
sey.generate_dict()

# Load data
cru = xr.open_dataset(dirs['ref'] + 'CRU/SECoW_CRU.nc')
cru = cru.reindex(source_cell=np.sort(cru.source_cell),
                  sink_cell=np.sort(cru.sink_cell))
src_cru = cru.src_CRU.values
snk_cru = cru.snk_CRU.values
tst = xr.open_dataset(dirs['ref'] + 'TST/SECoW_TST.nc').tst_cell
tst = tst.reindex(cell=np.sort(tst.cell)).values

# Process matrices
matrices = sey.process(fh='WINDS_coralsim_' + year + '*.nc', parameters=bio_data, RK4_its=RK4_its,
                       # Outputs
                       flux_src_grp=flux_src_grp, drift_time_grp=drift_time_grp,
                       src_str_cell=src_str_cell, snk_str_cell=snk_str_cell,
                       ret_str_cell=ret_str_cell, src_ent_cell=src_ent_cell,
                       snk_ent_cell=snk_ent_cell, flux_src_cell=flux_src_cell,
                       src_cru_cell=src_cru_cell, snk_cru_cell=snk_cru_cell,
                       src_tst_cell=src_tst_cell, snk_tst_cell=snk_tst_cell,
                       tst=tst, src_cru=src_cru, snk_cru=snk_cru)

for opt in opt_name_list.keys():
    if opt_name_list[opt]:
        matrix_fh = dirs['output'] + 'STEP_1/WINDS_' + opt + '_submatrix_' + bio_code + '_' + year + '.nc'
        matrices[opt].to_netcdf(matrix_fh, encoding={var: {'zlib': True, 'complevel': 5} for var in matrices[opt].variables})
