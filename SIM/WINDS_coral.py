#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run larval connectivity experiments in WINDS
@author: Noam Vogt-Vincent
"""

import os
from datetime import datetime, timedelta
from coralsim import Experiment
import sys

##############################################################################
# DEFINE INPUT FILES                                                         #
##############################################################################

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['model'] = dirs['root'] + 'MODEL_DATA/WINDS/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/'
dirs['traj'] = dirs['root'] + 'TRAJ/'

# PARAMETERS
year = int(sys.argv[1])
month = int(sys.argv[2])
day = int(sys.argv[3])
part = int(sys.argv[4])
partitions = int(sys.argv[5])

try:
    eez = int(sys.argv[6])
    eez_given = True
except:
    eez_given = False


pn_per_cell = 2**14
t0          = datetime(year=year,
                       month=month,
                       day=day, hour=0)
dt          = timedelta(minutes=10)
run_time    = timedelta(days=120)

##############################################################################
# SET UP EXPERIMENT                                                          #
##############################################################################

# Create experiment object
if eez_given:
    experiment = Experiment('WINDS_coralsim_' + str(year) + '_' + str(month) +
                            '_' + str(day) + '_' + str(part) + '_' + str(partitions)
                            + '_' + str(eez))
else:
    experiment = Experiment('WINDS_coralsim_' + str(year) + '_' + str(month) +
                            '_' + str(day) + '_' + str(part) + '_' + str(partitions))

# Run experiment
experiment.config(dirs, preset='WINDS')
experiment.generate_fieldset()
if eez_given:
   experiment.generate_particleset(num=pn_per_cell, t0=t0, filters={'eez': [eez]},
                                   min_competency=timedelta(days=0), dt=dt,
                                   run_time=run_time, partitions=partitions,
                                   part=part)
else:
    experiment.generate_particleset(num=pn_per_cell, t0=t0,
                                   min_competency=timedelta(days=0), dt=dt,
                                   run_time=run_time, partitions=partitions,
                                   part=part)
experiment.run()


