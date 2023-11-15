#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes and methods required to set up larval connectivity
experiments in OceanParcels.

@author: Noam Vogt-Vincent
@year: 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import cartopy.crs as ccrs
import xarray as xr
from glob import glob
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Variable)
from netCDF4 import Dataset
from datetime import timedelta, datetime
from tqdm import tqdm
from numba import njit
from time import time as systime

class Experiment():
    """
    Initialise a larval dispersal experiment.
    -----------
    Functions:
        # Preproc:
        config: register directories and load preset
        generate_fieldset: generate fieldsets for OceanParcels
        generate_particleset: generate initial conditions for particles + kernels

        # Runtime
        run: run OceanParcels using the above configuration

        # Not to be called:
        build_larva: build the larva class (used by generate_particleset)
        build_event: build the event kernel (used by generate_particleset)
    """


    def __init__(self, *args):
        # Set up a status dictionary so we know the completion status of the
        # experiment configuration

        self.status = {'config': False,
                       'fieldset': False,
                       'particleset': False,
                       'run': False,
                       'dict': False,
                       'matrix': False}

        try:
            self.name = args[0]
        except:
            self.name = 'my_experiment'


    def config(self, dir_dict, **kwargs):
        """
        Set directories for the script and import settings from preset

        Parameters
        ----------
        dir_dict : dictionary with 'root', 'grid', 'model', 'fig', and 'traj'
                   keys

        **kwargs : preset = 'CMEMS' or 'WINDS'

        """

        if not all (key in dir_dict.keys() for key in ['root', 'grid', 'model', 'fig', 'traj']):
            raise KeyError('Please make sure root, grid, model, fig, and traj directories have been specified.')

        if 'preset' not in kwargs.keys():
            raise NotImplementedError('Settings must currently be loaded from a preset.')

        # Note on grid types:
        # A-grid: RHO = U/V velocity defined here. Acts as 'edges' for cells.
        #         PSI = Must be calculated (as the midpoint between rho points).
        #               'Tracer' quantities (e.g. groups, coral cover, etc.)
        #               are defined here.
        # C-grid: RHO = 'Tracer' quantities (e.g. groups, coral cover, etc.)
        #               are defined here.
        #         U == (PSI, RHO) = U velocity defined here
        #         V == (RHO, PSI) = V velocity defined here

        # List all presets here
        CMEMS = {'preset': 'CMEMS',
                 'grid_filename': 'coral_grid.nc',
                 'model_filenames': 'CMEMS_SFC*.nc',

                 # Variable names for grid file
                 'grid_rc_varname' : 'reef_cover_c', # Reef cover
                 'grid_rf_varname' : 'reef_frac_c',  # Reef fraction
                 'grid_eez_varname': 'reef_eez_c',   # Reef EEZ
                 'grid_grp_varname': 'reef_grp_c',   # Reef group
                 'grid_idx_varname': 'reef_idx_c',   # Reef index,

                 # Variable types
                 'rc_dtype': np.int32,
                 'rf_dtype': np.float32,
                 'eez_dtype': np.int16,
                 'grp_dtype': np.uint8,
                 'idx_dtype': np.uint16,

                 # Dimension names for grid file
                 'lon_rho_dimname': 'lon_rho_c',
                 'lat_rho_dimname': 'lat_rho_c',
                 'lon_psi_dimname': 'lon_psi_c',
                 'lat_psi_dimname': 'lat_psi_c',

                 # Variable names for grid file
                 'u_varname': 'uo',
                 'v_varname': 'vo',

                 # Dimension names for model data
                 'lon_dimname': 'longitude',
                 'lat_dimname': 'latitude',
                 'time_dimname': 'time',

                 # Parameters for trajectory testing mode
                 'rel_lon0': 49.34,
                 'rel_lon1': 49.34,
                 'rel_lat0': -12.3,
                 'rel_lat1': -12.0,
                 'view_lon0': 47.5,
                 'view_lon1': 49.5,
                 'view_lat0': -13.5,
                 'view_lat1': -11.5,
                 'test_number': 100,
                 'lsm_varname': 'lsm_c',

                 # Grid type
                 'grid' : 'A',

                 # Maximum number of events
                 'e_num' : 36,

                 # Velocity interpolation method
                 'interp_method': 'freeslip',

                 # Plotting parameters
                 'plot': False,
                 'plot_type': 'grp',}

        WINDS = {'preset': 'WINDS',
                 'grid_filename': 'coral_grid.nc',
                 'model_filenames': 'WINDS_SFC*.nc',

                 # Variable names for grid file
                 'grid_rc_varname' : 'reef_cover_w', # Reef cover
                 'grid_rf_varname' : 'reef_frac_w',  # Reef fraction
                 'grid_eez_varname': 'reef_eez_w',   # Reef EEZ
                 'grid_grp_varname': 'reef_grp_w',   # Reef group
                 'grid_idx_varname': 'reef_idx_w',   # Reef index,

                 # Variable types
                 'rc_dtype': np.int32,
                 'rf_dtype': np.float32,
                 'eez_dtype': np.int16,
                 'grp_dtype': np.uint8,
                 'idx_dtype': np.uint16,

                 # Dimension names for grid file
                 'lon_rho_dimname': 'lon_rho_w',
                 'lat_rho_dimname': 'lat_rho_w',
                 'lon_psi_dimname': 'lon_psi_w',
                 'lat_psi_dimname': 'lat_psi_w',

                 # Variable names for grid file
                 'u_varname': 'u_surf',
                 'v_varname': 'v_surf',

                 # Dimension names for model data (psi-grid in this case)
                 'lon_dimname': 'nav_lon_u',
                 'lat_dimname': 'nav_lat_v',
                 'time_dimname': 'time_counter',

                 'lon_u_dimname': 'nav_lon_u',
                 'lat_u_dimname': 'nav_lat_u',
                 'lon_v_dimname': 'nav_lon_v',
                 'lat_v_dimname': 'nav_lat_v',

                 # Parameters for trajectory testing mode
                 'rel_lon0': 48.8,
                 'rel_lon1': 48.8,
                 'rel_lat0': -12.5,
                 'rel_lat1': -12.3,
                 'view_lon0': 48.6,
                 'view_lon1': 48.9,
                 'view_lat0': -12.5,
                 'view_lat1': -12.3,
                 'test_number': 5,
                 'lsm_varname': 'lsm_w',

                 # Grid type
                 'grid' : 'C',

                 # Maximum number of events
                 'e_num' : 60,

                 # Velocity interpolation method
                 'interp_method': 'cgrid_velocity',

                 # Plotting parameters
                 'plot': False,
                 'plot_type': 'grp',}

        PRESETS = {'CMEMS': CMEMS,
                   'WINDS': WINDS}

        if kwargs['preset'] not in PRESETS.keys():
            raise KeyError('Preset not recognised.')

        self.cfg = PRESETS[kwargs['preset']]
        self.dirs = dir_dict
        self.fh = {}

        # Further options
        if 'dt' in kwargs.keys():
            self.cfg['dt'] = kwargs['dt']

        if 'larvae_per_cell' in kwargs.keys():
            self.cfg['lpc'] = kwargs['larvae_per_cell']

        if 'releases_per_month' in kwargs.keys():
            self.cfg['rpm'] = kwargs['releases_per_month']

        if 'partitions' in kwargs.keys():
            self.cfg['partitions'] = kwargs['partitions']

        if 'test_params' in kwargs.keys():
            self.cfg['test_params'] = kwargs['test_params']


        def integrate_event(psi0, int0, fr, a, b, tc, μs, σ, λ, ν, t0, t1_prev, dt, its):
            # NOTE:
            # Due to the very large numbers used in intermediate steps, this
            # code currently runs in double precision, since the overflows
            # may occur with μs > ~1. If you are using this code for μs < 1
            # or you come up with a way of avoiding these very large
            # intermediates (I haven't found a way to do this in general),
            # this code can be accelerated by instead switching to single
            # precision.

            dt_ = dt/its
            sol = np.zeros_like(int0, dtype=np.float32)

            # Precompute constants for all iterations
            gc_0 = b-a+(μs*fr)
            gc_1 = μs*psi0

            f2_gc0 = np.exp(t0*(b-a))
            f3_gc0 = np.exp(t0*gc_0)

            f_2 = f2_gc0 - np.exp(t1_prev*(b-a))
            c_2 = np.exp(gc_1)/(b-a)
            c_3 = np.exp(gc_1-(μs*fr*t0))/gc_0

            int0_c2f2 = int0 + c_2*f_2

            for it in range(its):
                # Recompute initial time for iteration
                t0_ = t0 + (it/its)*dt

                # Integrate
                for h, rk_coef in zip(np.array([0, 0.5, 1], dtype=np.float32),
                                      np.array([1/6, 2/3, 1/6], dtype=np.float32)):

                    if h == 0:
                        t = t0_
                    else:
                        t = t0_ + h*dt_

                    if σ != 0:
                        surv_t = ((1. - σ*(λ*(t + tc))**ν)**(1/σ)).astype(np.float32)
                    else:
                        surv_t = np.exp(-(λ*(t + tc))**ν).astype(np.float32)

                    if h == 0 and it == 0:
                        f_3 = np.float32([0])
                        f_1 = surv_t*np.exp(-b*t)*np.exp(-μs*psi0)
                    else:
                        f_1 = surv_t*np.exp(-b*t)*np.exp(-μs*(psi0+(fr*(t-t0))))
                        f_3 = np.exp(t*gc_0) - f3_gc0

                    int1 = int0_c2f2 + c_3*f_3

                    sol += rk_coef*f_1*int1

            return a*μs*fr*dt_*sol, int1, psi0 + fr*dt

        self.integrate = integrate_event

        self.status['config'] = True


    def generate_fieldset(self, **kwargs):
        """
        Generate the FieldSet object for OceanParcels

        """

        if not self.status['config']:
            raise Exception('Please run config first.')

        # Generate file names
        self.fh['grid'] = self.dirs['grid'] + self.cfg['grid_filename']
        self.fh['model'] = sorted(glob(self.dirs['model'] + self.cfg['model_filenames']))

        # Import grid axes
        self.axes = {}

        with Dataset(self.fh['grid'], mode='r') as nc:
            self.axes['lon_rho'] = np.array(nc.variables[self.cfg['lon_rho_dimname']][:])
            self.axes['lat_rho'] = np.array(nc.variables[self.cfg['lat_rho_dimname']][:])
            self.axes['nx_rho'] = len(self.axes['lon_rho'])
            self.axes['ny_rho'] = len(self.axes['lat_rho'])

            self.axes['lon_psi'] = np.array(nc.variables[self.cfg['lon_psi_dimname']][:])
            self.axes['lat_psi'] = np.array(nc.variables[self.cfg['lat_psi_dimname']][:])
            self.axes['nx_psi'] = len(self.axes['lon_psi'])
            self.axes['ny_psi'] = len(self.axes['lat_psi'])

        # Import currents

        if self.cfg['grid'] == 'A':
            self.fieldset = FieldSet.from_netcdf(filenames=self.fh['model'],
                                                 variables={'U': self.cfg['u_varname'],
                                                            'V': self.cfg['v_varname']},
                                                 dimensions={'U': {'lon': self.cfg['lon_dimname'],
                                                                   'lat': self.cfg['lat_dimname'],
                                                                   'time': self.cfg['time_dimname']},
                                                             'V': {'lon': self.cfg['lon_dimname'],
                                                                   'lat': self.cfg['lat_dimname'],
                                                                   'time': self.cfg['time_dimname']}},
                                                 interp_method={'U': self.cfg['interp_method'],
                                                                'V': self.cfg['interp_method']},
                                                 mesh='spherical', allow_time_extrapolation=False)
        elif self.cfg['grid'] == 'C':
            self.fieldset = FieldSet.from_nemo(filenames=self.fh['model'],
                                               variables={'U': self.cfg['u_varname'],
                                                          'V': self.cfg['v_varname']},
                                               dimensions={'U': {'lon': self.cfg['lon_dimname'],
                                                                 'lat': self.cfg['lat_dimname'],
                                                                 'time': self.cfg['time_dimname']},
                                                           'V': {'lon': self.cfg['lon_dimname'],
                                                                 'lat': self.cfg['lat_dimname'],
                                                                 'time': self.cfg['time_dimname']}},
                                               mesh='spherical', allow_time_extrapolation=False)
        else:
            raise KeyError('Grid type not understood.')

        self.fields = {}

        # Import additional fields
        if self.cfg['grid'] == 'A':
            self.field_list = ['rc', 'rf', 'eez', 'grp', 'idx']

            for field in self.field_list:
                field_varname = self.cfg['grid_' + field + '_varname']

                # Firstly verify that dimensions are correct
                with Dataset(self.fh['grid'], mode='r') as nc:
                    self.fields[field] = nc.variables[field_varname][:]

                if not np.array_equiv(np.shape(self.fields[field]),
                                      (self.axes['ny_psi'], self.axes['nx_psi'])):
                    raise Exception('Field ' + field_varname + ' has incorrect dimensions')

                if field in ['rc', 'eez', 'grp', 'idx']:
                    if np.max(self.fields[field]) > np.iinfo(self.cfg[field + '_dtype']).max:
                        raise Exception('Maximum value exceeded in ' + field_varname + '.')

                # Use OceanParcels routine to import field
                scratch_field = Field.from_netcdf(self.fh['grid'],
                                                  variable=self.cfg['grid_' + str(field) + '_varname'],
                                                  dimensions={'lon': self.cfg['lon_psi_dimname'],
                                                              'lat': self.cfg['lat_psi_dimname']},
                                                  interp_method='nearest', mesh='spherical',
                                                  allow_time_extrapolation=True)

                scratch_field.name = field
                self.fieldset.add_field(scratch_field)

        elif self.cfg['grid'] == 'C':
            self.field_list = ['rc', 'rf', 'eez', 'grp', 'idx']

            for field in self.field_list:
                field_varname = self.cfg['grid_' + field + '_varname']

                # Firstly verify that dimensions are correct
                with Dataset(self.fh['grid'], mode='r') as nc:
                    self.fields[field] = nc.variables[field_varname][:]

                if not np.array_equiv(np.shape(self.fields[field]),
                                      (self.axes['ny_rho'], self.axes['nx_rho'])):
                    raise Exception('Field ' + field_varname + ' has incorrect dimensions')

                if field in ['rc', 'eez', 'grp', 'idx']:
                    if np.max(self.fields[field]) > np.iinfo(self.cfg[field + '_dtype']).max:
                        raise Exception('Maximum value exceeded in ' + field_varname + '.')

                # Use OceanParcels routine to import field
                scratch_field = Field.from_netcdf(self.fh['grid'],
                                                  variable=self.cfg['grid_' + str(field) + '_varname'],
                                                  dimensions={'lon': self.cfg['lon_rho_dimname'],
                                                              'lat': self.cfg['lat_rho_dimname']},
                                                  interp_method='nearest', mesh='spherical',
                                                  allow_time_extrapolation=True)
                scratch_field.name = field
                self.fieldset.add_field(scratch_field)

        self.status['fieldset'] = True


    def generate_particleset(self, **kwargs):

        """
        Generate the ParticleSet object for OceanParcels

        Parameters
        ----------
        **kwargs : num = Number of particles to (aim to) release per cell
                   filters = Dict with 'eez' and/or 'grp' keys to enable filter
                             for release sites

                   t0 = Release time for particles (datetime)
                   min_competency = Minimum competency period (timedelta)
                   dt = Model time-step (timedelta)
                   run_time = Model run-time (timedelta)
                   partitions = number of partitions to split pset into
                   part = which partition to choose (1, 2...)

                   test = Whether to activate testing kernels (bool)
        """

        if not self.status['fieldset']:
            raise Exception('Please run fieldset first.')

        # Generate required default values if necessary
        if 'num' not in kwargs.keys():
            print('Particle release number not supplied.')
            print('Setting to default of 100 per cell.')
            print('')
            self.cfg['pn'] = 10
        else:
            self.cfg['pn'] = int(np.ceil(kwargs['num']**0.5))
            self.cfg['pn2'] = int(self.cfg['pn']**2)

            if np.ceil(kwargs['num']**0.5) != kwargs['num']**0.5:
                print('Particle number per cell is not square.')
                print('Old particle number: ' + str(kwargs['num']))
                print('New particle number: ' + str(self.cfg['pn2']))
                print()

            if 'lpm' in self.cfg.keys():
                if self.cfg['pn2'] != self.cfg['lpm']:
                    raise Exception('Warning: lpm is inconsistent with existing particle number setting.')

        if 't0' not in kwargs.keys():
            print('Particle release time not supplied.')
            print('Setting to default of first time in file.')
            print('')
            self.cfg['t0'] = pd.Timestamp(self.fieldset.time_origin.time_origin)
        else:
            # Check that particle release time provided is not before first
            # available time in model data
            model_start = pd.Timestamp(self.fieldset.time_origin.time_origin)
            if pd.Timestamp(kwargs['t0']) < model_start:
                print(('Particle release time has been set to ' +
                       str(pd.Timestamp(kwargs['t0'])) +
                       ' but model data starts at ' +
                       str(model_start) + '. Shifting particle start to ' +
                       str(model_start) + '.'))

                self.cfg['t0'] = model_start

            else:
                self.cfg['t0'] = kwargs['t0']

        if 'filters'  in kwargs.keys():
            for filter_name in kwargs['filters'].keys():
                if filter_name not in ['eez', 'grp']:
                    raise KeyError('Filter name ' + filter_name + ' not understood.')
            filtering = True
            random_subset = False
        else:
            if 'n_sites' in kwargs.keys():
                assert kwargs['n_sites'] > 0
                filtering = True
                random_subset = True
            else:
                filtering = False
                random_subset = False

        if 'min_competency' in kwargs.keys():
            self.cfg['min_competency'] = kwargs['min_competency']
        else:
            print('Minimum competency period not supplied.')
            print('Setting to default of 2 days.')
            print('')
            self.cfg['min_competency'] = timedelta(days=2)

        if 'dt' in kwargs.keys():
            self.cfg['dt'] = kwargs['dt']
        else:
            print('RK4 timestep not supplied.')
            print('Setting to default of 1 hour.')
            print('')
            self.cfg['dt'] = timedelta(hours=1)

        if 'run_time' in kwargs.keys():
            self.cfg['run_time'] = kwargs['run_time']
        else:
            print('Run-time not supplied.')
            print('Setting to default of 100 days.')
            print('')
            self.cfg['run_time'] = timedelta(days=100)

        if 'test' in kwargs.keys():
            if kwargs['test'] in ['kernel', 'traj', 'vis', 'sens', False]:
                if kwargs['test'] in ['kernel', 'traj', 'vis', 'sens']:
                    self.cfg['test'] = True
                    self.cfg['test_type'] = kwargs['test']
                else:
                    self.cfg['test'] = False
                    self.cfg['test_type'] = False
            else:
                print('Test type not understood. Ignoring test.')
                self.cfg['test'] = False
                self.cfg['test_type'] = False
        else:
            self.cfg['test'] = False
            self.cfg['test_type'] = False

        if 'partitions' in kwargs.keys():
            if 'part' in kwargs.keys():
                self.cfg['partitions'] = kwargs['partitions']
                self.cfg['part'] = kwargs['part']
            else:
                raise Exception('Please specify which part of the partitionset to release.')
        else:
            self.cfg['partitions'] = False

        # Build a mask of valid initial position cells
        reef_mask = (self.fields['rc'] > 0)
        self.cfg['nsite_nofilter'] = int(np.sum(reef_mask))

        # Filter if applicable
        if filtering:
            if random_subset:
                idx_list = np.unique(self.fields['idx']).compressed()
                idx_list = idx_list[::int(np.ceil(len(idx_list)/kwargs['n_sites']))][:kwargs['n_sites']]
                reef_mask *= np.isin(self.fields['idx'], idx_list)
            else:
                for filter_name in kwargs['filters'].keys():
                    reef_mask *= np.isin(self.fields[filter_name], kwargs['filters'][filter_name])

        # Count number of sites identified
        self.cfg['nsite'] = int(np.sum(reef_mask))

        if self.cfg['nsite'] == 0:
            raise Exception('No valid reef sites found')
        else:
            print(str(self.cfg['nsite']) + '/' + str(self.cfg['nsite_nofilter'])  + ' reef sites identified.')
            print()

        # Find locations of sites
        reef_yidx, reef_xidx = np.where(reef_mask)

        # Generate meshgrids
        lon_rho_grid, lat_rho_grid = np.meshgrid(self.axes['lon_rho'],
                                                 self.axes['lat_rho'])
        lon_psi_grid, lat_psi_grid = np.meshgrid(self.axes['lon_psi'],
                                                 self.axes['lat_psi'])

        # Generate dictionary to hold initial particle properties
        particles = {}
        particles['lon'] = np.zeros((self.cfg['nsite']*self.cfg['pn2'],), dtype=np.float64)
        particles['lat'] = np.zeros((self.cfg['nsite']*self.cfg['pn2'],), dtype=np.float64)

        print(str(len(particles['lon'])) + ' particles generated.')
        print()

        for field in self.field_list:
            particles[field] = np.zeros((self.cfg['nsite']*self.cfg['pn2'],),
                                        dtype=self.cfg[field + '_dtype'])

        # Now evaluate each particle initial condition
        if self.cfg['grid'] == 'A':
            # For cell psi[i, j], the surrounding rho cells are:
            # rho[i, j]     (SW)
            # rho[i, j+1]   (SE)
            # rho[i+1, j]   (NW)
            # rho[i+1, j+1] (NE)

            for k, (i, j) in enumerate(zip(reef_yidx, reef_xidx)):
                # Firstly calculate the basic particle grid (may be variable for
                # curvilinear grids)

                dX = lon_rho_grid[i, j+1] - lon_rho_grid[i, j] # Grid spacing
                dY = lat_rho_grid[i+1, j] - lat_rho_grid[i, j] # Grid spacing
                dx = dX/self.cfg['pn']                         # Particle spacing
                dy = dY/self.cfg['pn']                         # Particle spacing

                gx = np.linspace(lon_rho_grid[i, j]+(dx/2),    # Particle x locations
                                 lon_rho_grid[i, j+1]-(dx/2), num=self.cfg['pn'])

                gy = np.linspace(lat_rho_grid[i, j]+(dy/2),    # Particle y locations
                                 lat_rho_grid[i+1, j]-(dy/2), num=self.cfg['pn'])

                gx, gy = [grid.flatten() for grid in np.meshgrid(gx, gy)] # Flattened arrays

                particles['lon'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gx
                particles['lat'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gy

                for field in self.field_list:
                    value_k = self.fields[field][i, j]
                    particles[field][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = value_k
        else:
            # For cell rho[i, j], the surrounding psi cells are:
            # psi[i-1, j-1] (SW)
            # psi[i-1, j]   (SE)
            # psi[i, j-1]   (NW)
            # psi[i, j]     (NE)

            for k, (i, j) in enumerate(zip(reef_yidx, reef_xidx)):
                # Firstly calculate the basic particle grid (may be variable for
                # curvilinear grids)

                dX = lon_psi_grid[i, j] - lon_psi_grid[i, j-1] # Grid spacing
                dY = lat_rho_grid[i, j] - lat_rho_grid[i-1, j] # Grid spacing
                dx = dX/self.cfg['pn']                         # Particle spacing
                dy = dY/self.cfg['pn']                         # Particle spacing

                gx = np.linspace(lon_rho_grid[i, j-1]+(dx/2),  # Particle x locations
                                 lon_rho_grid[i, j]-(dx/2), num=self.cfg['pn'])

                gy = np.linspace(lat_rho_grid[i-1, j]+(dy/2),  # Particle y locations
                                 lat_rho_grid[i, j]-(dy/2), num=self.cfg['pn'])

                gx, gy = [grid.flatten() for grid in np.meshgrid(gx, gy)] # Flattened arrays

                particles['lon'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gx
                particles['lat'][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = gy

                for field in self.field_list:
                    value_k = self.fields[field][i, j]
                    particles[field][k*self.cfg['pn2']:(k+1)*self.cfg['pn2']] = value_k

        # Now export to DataFrame
        particles_df = pd.DataFrame({'lon': particles['lon'],
                                     'lat': particles['lat']})

        for field in self.field_list:
            particles_df[field] = particles[field]

        # Now add release times
        particles_df['t0'] = self.cfg['t0']

        # Save particles_df to class
        self.particles = particles_df

        # Set up the particle class
        self.larva = self.build_larva(self.cfg['e_num'], self.cfg['test'], test_type=self.cfg['test_type'])

        # Override for the trajectory testing mode
        if self.cfg['test_type'] == 'traj':
            # Set t0 to first time frame
            self.cfg['t0'] = model_start

            # Override all properties with a smaller testing region
            particles['lon'] = np.linspace(self.cfg['rel_lon0'],
                                           self.cfg['rel_lon1'],
                                           num=self.cfg['test_number'])
            particles['lat'] = np.linspace(self.cfg['rel_lat0'],
                                           self.cfg['rel_lat1'],
                                           num=self.cfg['test_number'])

            self.particles = pd.DataFrame({'lon': particles['lon'],
                                           'lat': particles['lat'],
                                           't0': self.cfg['t0'],
                                           'idx': 1})

        if self.cfg['partitions']:
            self.particles = np.array_split(self.particles, self.cfg['partitions'])[self.cfg['part']-1]

        # Generate the ParticleSet
        if self.cfg['test_type'] != 'sens':
            self.pset = ParticleSet.from_list(fieldset=self.fieldset,
                                              pclass=self.larva,
                                              lonlatdepth_dtype=np.float64,
                                              lon=self.particles['lon'],
                                              lat=self.particles['lat'],
                                              time=self.particles['t0'],
                                              lon0=self.particles['lon'],
                                              lat0=self.particles['lat'],
                                              idx0=self.particles['idx'])
        else:
            self.pset = ParticleSet.from_list(fieldset=self.fieldset,
                                              pclass=self.larva,
                                              lonlatdepth_dtype=np.float64,
                                              lon=self.particles['lon'],
                                              lat=self.particles['lat'],
                                              time=self.particles['t0'],
                                              idx0=self.particles['idx'])

        # Stop writing unnecessary variables
        self.pset.set_variable_write_status('depth', 'False')
        self.pset.set_variable_write_status('time', 'False')

        if not self.cfg['test']:
            self.pset.set_variable_write_status('lon', 'False')
            self.pset.set_variable_write_status('lat', 'False')

        # Add maximum age to fieldset
        self.fieldset.add_constant('max_age', int(self.cfg['run_time']/self.cfg['dt']))
        assert self.fieldset.max_age < np.iinfo(np.uint16).max

        # Add e_num to fieldset
        self.fieldset.add_constant('e_num', int(self.cfg['e_num']))

        # Add test parameters to fieldset
        if self.cfg['test']:

            param_dict = {'a': 'a', 'b': 'b', 'tc': 'tc', 'μs': 'ms', 'σ': 'sig', 'ν': 'nu', 'λ': 'lam'}

            if 'test_params' not in self.cfg.keys():
                raise Exception('Test parameters not supplied.')

            for key in self.cfg['test_params'].keys():
                self.fieldset.add_constant(param_dict[key], self.cfg['test_params'][key])

            # In testing mode, we override the minimum competency to use tc
            self.fieldset.add_constant('min_competency', int(self.cfg['test_params']['tc']/self.cfg['dt'].total_seconds()))
        else:
            self.fieldset.add_constant('min_competency', int(self.cfg['min_competency']/self.cfg['dt']))

        # Generate kernels
        self.kernel = (self.pset.Kernel(AdvectionRK4) + self.pset.Kernel(self.build_event_kernel(self.cfg['test'], test_type=self.cfg['test_type'])))

        # Now plot initial conditions (if wished)
        if self.cfg['plot'] and not self.cfg['test']:
            colour_series = particles[self.cfg['plot_type']]

            plot_x_range = np.max(particles['lon']) - np.min(particles['lon'])
            plot_y_range = np.max(particles['lat']) - np.min(particles['lat'])
            plot_x_range = [np.min(particles['lon']) - 0.1*plot_x_range,
                            np.max(particles['lon']) + 0.1*plot_x_range]
            plot_y_range = [np.min(particles['lat']) - 0.1*plot_y_range,
                            np.max(particles['lat']) + 0.1*plot_y_range]
            aspect = (plot_y_range[1] - plot_y_range[0])/(plot_x_range[1] - plot_x_range[0])

            f, ax = plt.subplots(1, 1, figsize=(20, 20*aspect), subplot_kw={'projection': ccrs.PlateCarree()})
            cmap = 'prism'

            ax.set_xlim(plot_x_range)
            ax.set_ylim(plot_y_range)
            ax.set_title('Initial positions for particles')

            ax.scatter(particles['lon'], particles['lat'], c=colour_series,
                       cmap=cmap, s=1, transform=ccrs.PlateCarree())

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.8, color='black', linestyle='-', zorder=11)
            gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 5))
            gl.xlocator = mticker.FixedLocator(np.arange(0, 90, 5))
            gl.xlabels_top = False
            gl.ylabels_right = False

            plt.savefig(self.dirs['fig'] + 'initial_particle_position.png', dpi=300)
            plt.close()

        self.status['particleset'] = True

    def build_larva(self, e_num, test, **kwargs):
        """
        This script builds the larva class as a test or operational class based
        on whether test is True or False

        """

        if type(test) != bool:
            raise Exception('Input must be a boolean.')

        if test:
            if kwargs['test_type'] == 'vis':
                class larva(JITParticle):

                    ##################################################################
                    # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ######
                    ##################################################################

                    # idx of current cell (>0 if in any reef cell)
                    idx = Variable('idx',
                                   dtype=np.int32,
                                   initial=0,
                                   to_write=True)

                    # Time at sea (Total time steps since spawning)
                    ot  = Variable('ot',
                                   dtype=np.int32,
                                   initial=0,
                                   to_write=True)

                    # Active status
                    active = Variable('active',
                                      dtype=np.uint8,
                                      initial=1,
                                      to_write=False)

                    ##################################################################
                    # PROVENANCE IDENTIFIERS #########################################
                    ##################################################################

                    # Group of parent reef
                    idx0 = Variable('idx0',
                                    dtype=np.int32,
                                    to_write=True)

                    # Original longitude
                    lon0 = Variable('lon0',
                                    dtype=np.float32,
                                    to_write=True)

                    # Original latitude
                    lat0 = Variable('lat0',
                                    dtype=np.float32,
                                    to_write=True)

                    ##################################################################
                    # TEMPORARY VARIABLES FOR TRACKING SETTLING AT REEF SITES ########
                    ##################################################################

                    # Current reef time (record of timesteps spent at current reef cell)
                    # Switch to uint16 if possible!
                    current_reef_ts = Variable('current_reef_ts',
                                               dtype=np.int16,
                                               initial=0,
                                               to_write=False)

                    # Current reef t0 (record of arrival time (in timesteps) at current reef)
                    # Switch to uint16 if possible!
                    current_reef_ts0 = Variable('current_reef_ts0',
                                                dtype=np.int16,
                                                initial=0,
                                                to_write=False)

                    # Current reef idx (record of the index of the current reef
                    # Switch to uint16 if possible!
                    current_reef_idx = Variable('current_reef_idx',
                                                dtype=np.int32,
                                                initial=0.,
                                                to_write=False)

                    ##################################################################
                    # RECORD OF ALL EVENTS ###########################################
                    ##################################################################

                    # Number of events
                    e_num = Variable('e_num', dtype=np.int16, initial=0, to_write=True)

                    # Event variables (i = idx, t = arrival time(step), dt = time(steps) at reef)
                    # are added dynamically

                    ##################################################################
                    # TEMPORARY TESTING VARIABLES ####################################
                    ##################################################################

                    # Number of larvae accumulated in the current reef
                    L1 = Variable('L1', dtype=np.float64, initial=1., to_write=True) # Pre-competent larvae
                    L2 = Variable('L2', dtype=np.float64, initial=0., to_write=True) # Competent larvae
                    L10 = Variable('L10', dtype=np.float64, initial=0., to_write=True) # Pre-competent larvae, frozen at start
                    L20 = Variable('L20', dtype=np.float64, initial=0., to_write=True) # Competent larvae, frozen at start
                    Ns = Variable('Ns', dtype=np.float64, initial=0., to_write=True) # Larvae settling in current/just-passed event
                    Ns_next = Variable('Ns_next', dtype=np.float64, initial=0., to_write=True) # Larvae settling in current event (when event has just ended)

                    # Reef fraction
                    rf = Variable('rf', dtype=np.float32, initial=0., to_write=True)

                    # Mortality coefficient mu_m
                    mm = Variable('mm', dtype=np.float64, initial=0., to_write=True)

            elif kwargs['test_type'] == 'sens':
                class larva(JITParticle):

                    ##################################################################
                    # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ######
                    ##################################################################
                    # Time at sea (Total time steps since spawning)
                    ot  = Variable('ot',
                                   dtype=np.int32,
                                   initial=0,
                                   to_write=False)

                    # Active status
                    active = Variable('active',
                                      dtype=np.uint8,
                                      initial=1,
                                      to_write=False)

                    ##################################################################
                    # PROVENANCE IDENTIFIERS #########################################
                    ##################################################################

                    # Group of parent reef
                    idx0 = Variable('idx0',
                                    dtype=np.int32,
                                    to_write='once')

                    ##################################################################
                    # TESTING VARIABLES ##############################################
                    ##################################################################

                    L1 = Variable('L1', dtype=np.float64, initial=1., to_write=False) # Pre-competent larvae
                    L2 = Variable('L2', dtype=np.float64, initial=0., to_write=False) # Competent larvae
                    L10 = Variable('L10', dtype=np.float64, initial=0., to_write=False) # Pre-competent larvae, frozen at start
                    L20 = Variable('L20', dtype=np.float64, initial=0., to_write=False) # Competent larvae, frozen at startended)
                    Ns = Variable('Ns', dtype=np.float32, initial=0., to_write=True) # Total settled larvae

                    # Reef fraction
                    rf = Variable('rf', dtype=np.float32, initial=0., to_write=False)

                    # Mortality coefficient mu_m
                    mm = Variable('mm', dtype=np.float64, initial=0., to_write=False)

            else:
                class larva(JITParticle):

                    ##################################################################
                    # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ######
                    ##################################################################

                    # idx of current cell (>0 if in any reef cell)
                    idx = Variable('idx',
                                   dtype=np.int32,
                                   initial=0,
                                   to_write=False)

                    # Time at sea (Total time steps since spawning)
                    ot  = Variable('ot',
                                   dtype=np.int32,
                                   initial=0,
                                   to_write=False)

                    # Active status
                    active = Variable('active',
                                      dtype=np.uint8,
                                      initial=1,
                                      to_write=False)

                    ##################################################################
                    # PROVENANCE IDENTIFIERS #########################################
                    ##################################################################

                    # Group of parent reef
                    idx0 = Variable('idx0',
                                    dtype=np.int32,
                                    to_write='once')

                    # Original longitude
                    lon0 = Variable('lon0',
                                    dtype=np.float32,
                                    to_write=False)

                    # Original latitude
                    lat0 = Variable('lat0',
                                    dtype=np.float32,
                                    to_write=False)

                    ##################################################################
                    # TEMPORARY VARIABLES FOR TRACKING SETTLING AT REEF SITES ########
                    ##################################################################

                    # Current reef time (record of timesteps spent at current reef cell)
                    # Switch to uint16 if possible!
                    current_reef_ts = Variable('current_reef_ts',
                                               dtype=np.int16,
                                               initial=0,
                                               to_write=False)

                    # Current reef t0 (record of arrival time (in timesteps) at current reef)
                    # Switch to uint16 if possible!
                    current_reef_ts0 = Variable('current_reef_ts0',
                                                dtype=np.int16,
                                                initial=0,
                                                to_write=False)

                    # Current reef idx (record of the index of the current reef
                    # Switch to uint16 if possible!
                    current_reef_idx = Variable('current_reef_idx',
                                                dtype=np.int32,
                                                initial=0.,
                                                to_write=False)

                    ##################################################################
                    # RECORD OF ALL EVENTS ###########################################
                    ##################################################################

                    # Number of events
                    e_num = Variable('e_num', dtype=np.int16, initial=0, to_write=False)

                    # Event variables (i = idx, t = arrival time(step), dt = time(steps) at reef)
                    # are added dynamically

                    ##################################################################
                    # TEMPORARY TESTING VARIABLES ####################################
                    ##################################################################

                    # Number of larvae accumulated in the current reef
                    L1 = Variable('L1', dtype=np.float64, initial=1., to_write=False) # Pre-competent larvae
                    L2 = Variable('L2', dtype=np.float64, initial=0., to_write=True) # Competent larvae
                    L10 = Variable('L10', dtype=np.float64, initial=0., to_write=False) # Pre-competent larvae, frozen at start
                    L20 = Variable('L20', dtype=np.float64, initial=0., to_write=False) # Competent larvae, frozen at start
                    Ns = Variable('Ns', dtype=np.float64, initial=0., to_write=False) # Settled larvae

                    # Reef fraction
                    rf = Variable('rf', dtype=np.float32, initial=0., to_write=False)

                    # Mortality coefficient mu_m
                    mm = Variable('mm', dtype=np.float64, initial=0., to_write=False)

        else:
            class larva(JITParticle):

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ######
                ##################################################################

                # idx of current cell (>0 if in any reef cell)
                idx = Variable('idx',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Time at sea (Total time steps since spawning)
                ot  = Variable('ot',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Active status
                active = Variable('active',
                                  dtype=np.uint8,
                                  initial=1,
                                  to_write=False)

                ##################################################################
                # PROVENANCE IDENTIFIERS #########################################
                ##################################################################

                # Group of parent reef
                idx0 = Variable('idx0',
                                dtype=np.uint16,
                                to_write=True)

                # Original longitude
                lon0 = Variable('lon0',
                                dtype=np.float32,
                                to_write=True)

                # Original latitude
                lat0 = Variable('lat0',
                                dtype=np.float32,
                                to_write=True)

                ##################################################################
                # TEMPORARY VARIABLES FOR TRACKING SETTLING AT REEF SITES ########
                ##################################################################

                # Current reef time (record of timesteps spent at current reef cell)
                # Switch to uint16 if possible!
                current_reef_ts = Variable('current_reef_ts',
                                           dtype=np.uint16,
                                           initial=0,
                                           to_write=False)

                # Current reef t0 (record of arrival time (in timesteps) at current reef)
                # Switch to uint16 if possible!
                current_reef_ts0 = Variable('current_reef_ts0',
                                            dtype=np.uint16,
                                            initial=0,
                                            to_write=False)

                # Current reef idx (record of the index of the current reef
                # Switch to uint16 if possible!
                current_reef_idx = Variable('current_reef_idx',
                                            dtype=np.uint16,
                                            initial=0.,
                                            to_write=False)

                ##################################################################
                # RECORD OF ALL EVENTS ###########################################
                ##################################################################

                # Number of events
                e_num = Variable('e_num', dtype=np.uint8, initial=0, to_write=True)

                # Event variables (i = idx, t = arrival time(step), dt = time(steps) at reef)
                # are added dynamically

        if not test or (kwargs['test_type'] not in ['vis', 'sens']):
            for e_val in range(e_num):
                setattr(larva, 'i' + str(e_val), Variable('i' + str(e_val), dtype=np.uint16, initial=0, to_write=True))
                setattr(larva, 'ts' + str(e_val), Variable('ts' + str(e_val), dtype=np.uint16, initial=0, to_write=True))
                setattr(larva, 'dt' + str(e_val), Variable('dt' + str(e_val), dtype=np.uint16, initial=0, to_write=True))

                if test:
                    setattr(larva, 'Ns' + str(e_val), Variable('Ns' + str(e_val), dtype=np.float32, initial=0., to_write=True))

        return larva

    def build_event_kernel(self, test, **kwargs):
        """
        This script builds the event kernel as a test or operational kernel based
        on whether test is True or False

        """

        if type(test) != bool:
            raise Exception('Input must be a boolean.')

        if test:
            if kwargs['test_type'] == 'vis':
                def event(particle, fieldset, time):

                    # 1 Keep track of the amount of time spent at sea
                    particle.ot += 1

                    # 2 Assess reef status
                    particle.idx = fieldset.idx[particle]

                    # Calculate current mortality rate
                    particle.mm = (fieldset.lam*fieldset.nu)*((fieldset.lam*particle.ot*particle.dt)**(fieldset.nu-1))/(1-fieldset.sig*((fieldset.lam*particle.ot*particle.dt)**fieldset.nu))
                    particle.L10 = particle.L1
                    particle.L20 = particle.L2

                    particle.rf = fieldset.rf[particle]

                    # 3 Trigger event cascade if larva is in a reef site and minimum competency has been reached
                    if particle.idx > 0 and particle.ot > fieldset.min_competency:

                        particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                        particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt

                    elif particle.ot > fieldset.min_competency:

                        particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                        particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm)*particle.L20)*particle.dt

                    else:
                        particle.L1 = particle.L10 - (particle.mm)*particle.L10*particle.dt

            elif kwargs['test_type'] == 'sens':
                def event(particle, fieldset, time):

                    # 1 Keep track of the amount of time spent at sea
                    particle.ot += 1

                    # 2 Assess reef status
                    particle.rf = fieldset.rf[particle]

                    # Calculate current mortality rate
                    particle.mm = (fieldset.lam*fieldset.nu)*((fieldset.lam*particle.ot*particle.dt)**(fieldset.nu-1))/(1-fieldset.sig*((fieldset.lam*particle.ot*particle.dt)**fieldset.nu))
                    particle.L10 = particle.L1
                    particle.L20 = particle.L2

                    # 3 Trigger event cascade if larva is in a reef site and minimum competency has been reached
                    if particle.rf > 0 and particle.ot > fieldset.min_competency:

                        particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                        particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                        particle.Ns = particle.Ns + fieldset.ms*particle.rf*particle.L20*particle.dt

                    elif particle.ot > fieldset.min_competency:

                        particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                        particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm)*particle.L20)*particle.dt

                    else:
                        particle.L1 = particle.L10 - (particle.mm)*particle.L10*particle.dt
            else:
                def event(particle, fieldset, time):

                    # 1 Keep track of the amount of time spent at sea
                    particle.ot += 1

                    ###############################################################
                    # ACTIVE PARTICLES ONLY                                       #
                    ###############################################################

                    if particle.active:

                        # 2 Assess reef status
                        particle.idx = fieldset.idx[particle]

                        # TESTING ONLY ############################################
                        # Calculate current mortality rate
                        particle.mm = (fieldset.lam*fieldset.nu)*((fieldset.lam*particle.ot*particle.dt)**(fieldset.nu-1))/(1-fieldset.sig*((fieldset.lam*particle.ot*particle.dt)**fieldset.nu))
                        particle.L10 = particle.L1
                        particle.L20 = particle.L2

                        particle.rf = fieldset.rf[particle]
                        ###########################################################

                        save_event = False
                        new_event = False

                        # 3 Trigger event cascade if larva is in a reef site and minimum competency has been reached
                        if particle.idx > 0 and particle.ot > fieldset.min_competency:

                            # Check if an event has already been triggered
                            if particle.current_reef_ts > 0:

                                # Check if we are in the same reef idx as the current event
                                if particle.idx == particle.current_reef_idx:

                                    # If contiguous event, just add time and phi
                                    particle.current_reef_ts += 1

                                    # TESTING ONLY ############################################
                                    particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                                    particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                                    particle.Ns = particle.Ns + fieldset.ms*particle.rf*particle.L20*particle.dt
                                    ###########################################################

                                    # But also check that the particle isn't about to expire (save if so)
                                    # Otherwise particles hanging around reefs at the end of the simulation
                                    # won't get saved.

                                    if particle.ot > fieldset.max_age:
                                        save_event = True

                                else:

                                    # TESTING ONLY ############################################
                                    particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                                    particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                                    particle.Ns = particle.Ns
                                    particle.Ns_next = fieldset.ms*particle.rf*particle.L20*particle.dt
                                    ###########################################################

                                    # Otherwise, we need to save the old event and create a new event
                                    save_event = True
                                    new_event = True

                            else:

                                # TESTING ONLY ############################################
                                particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                                particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                                particle.Ns_next = fieldset.ms*particle.rf*particle.L20*particle.dt
                                ###########################################################

                                # If event has not been triggered, create a new event
                                new_event = True

                        else:

                            # Otherwise, check if ongoing event has just ended
                            if particle.current_reef_ts > 0 and particle.ot > fieldset.min_competency:

                                # TESTING ONLY ############################################
                                particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                                particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm + fieldset.ms*particle.rf)*particle.L20)*particle.dt
                                particle.Ns = particle.Ns + fieldset.ms*particle.rf*particle.L20*particle.dt
                                ###########################################################

                                save_event = True

                            elif particle.ot > fieldset.min_competency:
                                # TESTING ONLY ############################################
                                particle.L1 = particle.L10 - (fieldset.a + particle.mm)*particle.L10*particle.dt
                                particle.L2 = particle.L20 + ((fieldset.a*particle.L10) - (fieldset.b + particle.mm)*particle.L20)*particle.dt
                                ###########################################################

                            else:
                                # TESTING ONLY ############################################
                                particle.L1 = particle.L10 - (particle.mm)*particle.L10*particle.dt
                                ###########################################################


                        if save_event:
                            # Save current values
                            # Unfortunately since setattr doesn't work in a kernel, this
                            # requires a horrendous elif chain.

                            if particle.e_num == 0:
                                particle.i0 = particle.current_reef_idx
                                particle.ts0 = particle.current_reef_ts0
                                particle.dt0 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns0 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 1:
                                particle.i1 = particle.current_reef_idx
                                particle.ts1 = particle.current_reef_ts0
                                particle.dt1 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns1 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 2:
                                particle.i2 = particle.current_reef_idx
                                particle.ts2 = particle.current_reef_ts0
                                particle.dt2 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns2 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 3:
                                particle.i3 = particle.current_reef_idx
                                particle.ts3 = particle.current_reef_ts0
                                particle.dt3 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns3 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 4:
                                particle.i4 = particle.current_reef_idx
                                particle.ts4 = particle.current_reef_ts0
                                particle.dt4 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns4 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 5:
                                particle.i5 = particle.current_reef_idx
                                particle.ts5 = particle.current_reef_ts0
                                particle.dt5 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns5 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 6:
                                particle.i6 = particle.current_reef_idx
                                particle.ts6 = particle.current_reef_ts0
                                particle.dt6 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns6 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 7:
                                particle.i7 = particle.current_reef_idx
                                particle.ts7 = particle.current_reef_ts0
                                particle.dt7 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns7 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 8:
                                particle.i8 = particle.current_reef_idx
                                particle.ts8 = particle.current_reef_ts0
                                particle.dt8 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns8 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 9:
                                particle.i9 = particle.current_reef_idx
                                particle.ts9 = particle.current_reef_ts0
                                particle.dt9 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns9 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 10:
                                particle.i10 = particle.current_reef_idx
                                particle.ts10 = particle.current_reef_ts0
                                particle.dt10 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns10 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 11:
                                particle.i11 = particle.current_reef_idx
                                particle.ts11 = particle.current_reef_ts0
                                particle.dt11 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns11 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 12:
                                particle.i12 = particle.current_reef_idx
                                particle.ts12 = particle.current_reef_ts0
                                particle.dt12 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns12 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 13:
                                particle.i13 = particle.current_reef_idx
                                particle.ts13 = particle.current_reef_ts0
                                particle.dt13 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns13 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 14:
                                particle.i14 = particle.current_reef_idx
                                particle.ts14 = particle.current_reef_ts0
                                particle.dt14 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns14 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 15:
                                particle.i15 = particle.current_reef_idx
                                particle.ts15 = particle.current_reef_ts0
                                particle.dt15 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns15 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 16:
                                particle.i16 = particle.current_reef_idx
                                particle.ts16 = particle.current_reef_ts0
                                particle.dt16 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns16 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 17:
                                particle.i17 = particle.current_reef_idx
                                particle.ts17 = particle.current_reef_ts0
                                particle.dt17 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns17 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 18:
                                particle.i18 = particle.current_reef_idx
                                particle.ts18 = particle.current_reef_ts0
                                particle.dt18 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns18 = particle.Ns
                                ###########################################################
                            elif particle.e_num == 19:
                                particle.i19 = particle.current_reef_idx
                                particle.ts19 = particle.current_reef_ts0
                                particle.dt19 = particle.current_reef_ts
                                # TESTING ONLY ############################################
                                particle.Ns19 = particle.Ns
                                ###########################################################

                                particle.active = 0 # Deactivate particle, since no more reefs can be saved

                            # Then reset current values to zero
                            particle.current_reef_idx = 0
                            particle.current_reef_ts0 = 0
                            particle.current_reef_ts = 0

                            # Add to event number counter
                            particle.e_num += 1

                        if new_event:
                            # Add status to current (for current event) values
                            # Timesteps at current reef
                            particle.current_reef_ts = 1

                            # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                            particle.current_reef_ts0 = particle.ot - 1

                            # Current reef group
                            particle.current_reef_idx = particle.idx

                            # TESTING ONLY ############################################
                            particle.Ns = particle.Ns_next
                            ###########################################################

                    # Finally, check if particle needs to be deleted
                    if particle.ot >= fieldset.max_age:

                        # Only delete particles where at least 1 event has been recorded
                        if particle.e_num > 0:
                            particle.delete()
        else:
            if self.cfg['preset'] == 'CMEMS':
                def event(particle, fieldset, time):

                    # 1 Keep track of the amount of time spent at sea
                    particle.ot += 1

                    ###############################################################
                    # ACTIVE PARTICLES ONLY                                       #
                    ###############################################################

                    if particle.active:

                        # 2 Assess reef status
                        particle.idx = fieldset.idx[particle]

                        save_event = False
                        new_event = False

                        # 3 Trigger event cascade if larva is in a reef site and competency has been reached
                        if particle.idx > 0 and particle.ot > fieldset.min_competency:

                            # Check if an event has already been triggered
                            if particle.current_reef_ts > 0:

                                # Check if we are in the same reef idx as the current event
                                if particle.idx == particle.current_reef_idx:

                                    # If contiguous event, just add time and phi
                                    particle.current_reef_ts += 1

                                    # But also check that the particle isn't about to expire (save if so)
                                    # Otherwise particles hanging around reefs at the end of the simulation
                                    # won't get saved.

                                    if particle.ot > fieldset.max_age:
                                        save_event = True

                                else:

                                    # Otherwise, we need to save the old event and create a new event
                                    save_event = True
                                    new_event = True

                            else:

                                # If event has not been triggered, create a new event
                                new_event = True

                        else:

                            # Otherwise, check if ongoing event has just ended
                            if particle.current_reef_ts > 0:

                                save_event = True

                        if save_event:
                            # Save current values
                            # Unfortunately, due to the limited functions allowed in parcels, this
                            # required an horrendous if-else chain

                            if particle.e_num == 0:
                                particle.i0 = particle.current_reef_idx
                                particle.ts0 = particle.current_reef_ts0
                                particle.dt0 = particle.current_reef_ts
                            elif particle.e_num == 1:
                                particle.i1 = particle.current_reef_idx
                                particle.ts1 = particle.current_reef_ts0
                                particle.dt1 = particle.current_reef_ts
                            elif particle.e_num == 2:
                                particle.i2 = particle.current_reef_idx
                                particle.ts2 = particle.current_reef_ts0
                                particle.dt2 = particle.current_reef_ts
                            elif particle.e_num == 3:
                                particle.i3 = particle.current_reef_idx
                                particle.ts3 = particle.current_reef_ts0
                                particle.dt3 = particle.current_reef_ts
                            elif particle.e_num == 4:
                                particle.i4 = particle.current_reef_idx
                                particle.ts4 = particle.current_reef_ts0
                                particle.dt4 = particle.current_reef_ts
                            elif particle.e_num == 5:
                                particle.i5 = particle.current_reef_idx
                                particle.ts5 = particle.current_reef_ts0
                                particle.dt5 = particle.current_reef_ts
                            elif particle.e_num == 6:
                                particle.i6 = particle.current_reef_idx
                                particle.ts6 = particle.current_reef_ts0
                                particle.dt6 = particle.current_reef_ts
                            elif particle.e_num == 7:
                                particle.i7 = particle.current_reef_idx
                                particle.ts7 = particle.current_reef_ts0
                                particle.dt7 = particle.current_reef_ts
                            elif particle.e_num == 8:
                                particle.i8 = particle.current_reef_idx
                                particle.ts8 = particle.current_reef_ts0
                                particle.dt8 = particle.current_reef_ts
                            elif particle.e_num == 9:
                                particle.i9 = particle.current_reef_idx
                                particle.ts9 = particle.current_reef_ts0
                                particle.dt9 = particle.current_reef_ts
                            elif particle.e_num == 10:
                                particle.i10 = particle.current_reef_idx
                                particle.ts10 = particle.current_reef_ts0
                                particle.dt10 = particle.current_reef_ts
                            elif particle.e_num == 11:
                                particle.i11 = particle.current_reef_idx
                                particle.ts11 = particle.current_reef_ts0
                                particle.dt11 = particle.current_reef_ts
                            elif particle.e_num == 12:
                                particle.i12 = particle.current_reef_idx
                                particle.ts12 = particle.current_reef_ts0
                                particle.dt12 = particle.current_reef_ts
                            elif particle.e_num == 13:
                                particle.i13 = particle.current_reef_idx
                                particle.ts13 = particle.current_reef_ts0
                                particle.dt13 = particle.current_reef_ts
                            elif particle.e_num == 14:
                                particle.i14 = particle.current_reef_idx
                                particle.ts14 = particle.current_reef_ts0
                                particle.dt14 = particle.current_reef_ts
                            elif particle.e_num == 15:
                                particle.i15 = particle.current_reef_idx
                                particle.ts15 = particle.current_reef_ts0
                                particle.dt15 = particle.current_reef_ts
                            elif particle.e_num == 16:
                                particle.i16 = particle.current_reef_idx
                                particle.ts16 = particle.current_reef_ts0
                                particle.dt16 = particle.current_reef_ts
                            elif particle.e_num == 17:
                                particle.i17 = particle.current_reef_idx
                                particle.ts17 = particle.current_reef_ts0
                                particle.dt17 = particle.current_reef_ts
                            elif particle.e_num == 18:
                                particle.i18 = particle.current_reef_idx
                                particle.ts18 = particle.current_reef_ts0
                                particle.dt18 = particle.current_reef_ts
                            elif particle.e_num == 19:
                                particle.i19 = particle.current_reef_idx
                                particle.ts19 = particle.current_reef_ts0
                                particle.dt19 = particle.current_reef_ts
                            elif particle.e_num == 20:
                                particle.i20 = particle.current_reef_idx
                                particle.ts20 = particle.current_reef_ts0
                                particle.dt20 = particle.current_reef_ts
                            elif particle.e_num == 21:
                                particle.i21 = particle.current_reef_idx
                                particle.ts21 = particle.current_reef_ts0
                                particle.dt21 = particle.current_reef_ts
                            elif particle.e_num == 22:
                                particle.i22 = particle.current_reef_idx
                                particle.ts22 = particle.current_reef_ts0
                                particle.dt22 = particle.current_reef_ts
                            elif particle.e_num == 23:
                                particle.i23 = particle.current_reef_idx
                                particle.ts23 = particle.current_reef_ts0
                                particle.dt23 = particle.current_reef_ts
                            elif particle.e_num == 24:
                                particle.i24 = particle.current_reef_idx
                                particle.ts24 = particle.current_reef_ts0
                                particle.dt24 = particle.current_reef_ts
                            elif particle.e_num == 25:
                                particle.i25 = particle.current_reef_idx
                                particle.ts25 = particle.current_reef_ts0
                                particle.dt25 = particle.current_reef_ts
                            elif particle.e_num == 26:
                                particle.i26 = particle.current_reef_idx
                                particle.ts26 = particle.current_reef_ts0
                                particle.dt26 = particle.current_reef_ts
                            elif particle.e_num == 27:
                                particle.i27 = particle.current_reef_idx
                                particle.ts27 = particle.current_reef_ts0
                                particle.dt27 = particle.current_reef_ts
                            elif particle.e_num == 28:
                                particle.i28 = particle.current_reef_idx
                                particle.ts28 = particle.current_reef_ts0
                                particle.dt28 = particle.current_reef_ts
                            elif particle.e_num == 29:
                                particle.i29 = particle.current_reef_idx
                                particle.ts29 = particle.current_reef_ts0
                                particle.dt29 = particle.current_reef_ts
                            elif particle.e_num == 30:
                                particle.i30 = particle.current_reef_idx
                                particle.ts30 = particle.current_reef_ts0
                                particle.dt30 = particle.current_reef_ts
                            elif particle.e_num == 31:
                                particle.i31 = particle.current_reef_idx
                                particle.ts31 = particle.current_reef_ts0
                                particle.dt31 = particle.current_reef_ts
                            elif particle.e_num == 32:
                                particle.i32 = particle.current_reef_idx
                                particle.ts32 = particle.current_reef_ts0
                                particle.dt32 = particle.current_reef_ts
                            elif particle.e_num == 33:
                                particle.i33 = particle.current_reef_idx
                                particle.ts33 = particle.current_reef_ts0
                                particle.dt33 = particle.current_reef_ts
                            elif particle.e_num == 34:
                                particle.i34 = particle.current_reef_idx
                                particle.ts34 = particle.current_reef_ts0
                                particle.dt34 = particle.current_reef_ts
                            elif particle.e_num == 35:
                                particle.i35 = particle.current_reef_idx
                                particle.ts35 = particle.current_reef_ts0
                                particle.dt35 = particle.current_reef_ts

                                particle.active = 0  # Deactivate particle, since no more reefs can be saved

                            # Then reset current values to zero
                            particle.current_reef_idx = 0
                            particle.current_reef_ts0 = 0
                            particle.current_reef_ts = 0

                            # Add to event number counter
                            particle.e_num += 1

                        if new_event:
                            # Add status to current (for current event) values
                            # Timesteps at current reef
                            particle.current_reef_ts = 1

                            # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                            particle.current_reef_ts0 = particle.ot - 1

                            # Current reef group
                            particle.current_reef_idx = particle.idx

                    # Finally, check if particle needs to be deleted
                    if particle.ot >= fieldset.max_age:

                        # Only delete particles where at least 1 event has been recorded
                        if particle.e_num > 0:
                            particle.delete()

            elif self.cfg['preset'] == 'WINDS':
                # OPERATIONAL WINDS KERNEL
                def event(particle, fieldset, time):

                    # 1 Keep track of the amount of time spent at sea
                    particle.ot += 1

                    ###############################################################
                    # ACTIVE PARTICLES ONLY                                       #
                    ###############################################################

                    if particle.active:

                        # 2 Assess reef status
                        particle.idx = fieldset.idx[particle]

                        save_event = False
                        new_event = False

                        # 3 Trigger event cascade if larva is in a reef site and competency has been reached
                        if particle.idx > 0 and particle.ot > fieldset.min_competency:

                            # Check if an event has already been triggered
                            if particle.current_reef_ts > 0:

                                # Check if we are in the same reef idx as the current event
                                if particle.idx == particle.current_reef_idx:

                                    # If contiguous event, just add time and phi
                                    particle.current_reef_ts += 1

                                    # But also check that the particle isn't about to expire (save if so)
                                    # Otherwise particles hanging around reefs at the end of the simulation
                                    # won't get saved.

                                    if particle.ot > fieldset.max_age:
                                        save_event = True

                                else:

                                    # Otherwise, we need to save the old event and create a new event
                                    save_event = True
                                    new_event = True

                            else:

                                # If event has not been triggered, create a new event
                                new_event = True

                        else:

                            # Otherwise, check if ongoing event has just ended
                            if particle.current_reef_ts > 0:

                                save_event = True

                        if save_event:
                            # Save current values
                            # Unfortunately, due to the limited functions allowed in parcels, this
                            # required an horrendous if-else chain

                            if particle.e_num == 0:
                                particle.i0 = particle.current_reef_idx
                                particle.ts0 = particle.current_reef_ts0
                                particle.dt0 = particle.current_reef_ts
                            elif particle.e_num == 1:
                                particle.i1 = particle.current_reef_idx
                                particle.ts1 = particle.current_reef_ts0
                                particle.dt1 = particle.current_reef_ts
                            elif particle.e_num == 2:
                                particle.i2 = particle.current_reef_idx
                                particle.ts2 = particle.current_reef_ts0
                                particle.dt2 = particle.current_reef_ts
                            elif particle.e_num == 3:
                                particle.i3 = particle.current_reef_idx
                                particle.ts3 = particle.current_reef_ts0
                                particle.dt3 = particle.current_reef_ts
                            elif particle.e_num == 4:
                                particle.i4 = particle.current_reef_idx
                                particle.ts4 = particle.current_reef_ts0
                                particle.dt4 = particle.current_reef_ts
                            elif particle.e_num == 5:
                                particle.i5 = particle.current_reef_idx
                                particle.ts5 = particle.current_reef_ts0
                                particle.dt5 = particle.current_reef_ts
                            elif particle.e_num == 6:
                                particle.i6 = particle.current_reef_idx
                                particle.ts6 = particle.current_reef_ts0
                                particle.dt6 = particle.current_reef_ts
                            elif particle.e_num == 7:
                                particle.i7 = particle.current_reef_idx
                                particle.ts7 = particle.current_reef_ts0
                                particle.dt7 = particle.current_reef_ts
                            elif particle.e_num == 8:
                                particle.i8 = particle.current_reef_idx
                                particle.ts8 = particle.current_reef_ts0
                                particle.dt8 = particle.current_reef_ts
                            elif particle.e_num == 9:
                                particle.i9 = particle.current_reef_idx
                                particle.ts9 = particle.current_reef_ts0
                                particle.dt9 = particle.current_reef_ts
                            elif particle.e_num == 10:
                                particle.i10 = particle.current_reef_idx
                                particle.ts10 = particle.current_reef_ts0
                                particle.dt10 = particle.current_reef_ts
                            elif particle.e_num == 11:
                                particle.i11 = particle.current_reef_idx
                                particle.ts11 = particle.current_reef_ts0
                                particle.dt11 = particle.current_reef_ts
                            elif particle.e_num == 12:
                                particle.i12 = particle.current_reef_idx
                                particle.ts12 = particle.current_reef_ts0
                                particle.dt12 = particle.current_reef_ts
                            elif particle.e_num == 13:
                                particle.i13 = particle.current_reef_idx
                                particle.ts13 = particle.current_reef_ts0
                                particle.dt13 = particle.current_reef_ts
                            elif particle.e_num == 14:
                                particle.i14 = particle.current_reef_idx
                                particle.ts14 = particle.current_reef_ts0
                                particle.dt14 = particle.current_reef_ts
                            elif particle.e_num == 15:
                                particle.i15 = particle.current_reef_idx
                                particle.ts15 = particle.current_reef_ts0
                                particle.dt15 = particle.current_reef_ts
                            elif particle.e_num == 16:
                                particle.i16 = particle.current_reef_idx
                                particle.ts16 = particle.current_reef_ts0
                                particle.dt16 = particle.current_reef_ts
                            elif particle.e_num == 17:
                                particle.i17 = particle.current_reef_idx
                                particle.ts17 = particle.current_reef_ts0
                                particle.dt17 = particle.current_reef_ts
                            elif particle.e_num == 18:
                                particle.i18 = particle.current_reef_idx
                                particle.ts18 = particle.current_reef_ts0
                                particle.dt18 = particle.current_reef_ts
                            elif particle.e_num == 19:
                                particle.i19 = particle.current_reef_idx
                                particle.ts19 = particle.current_reef_ts0
                                particle.dt19 = particle.current_reef_ts
                            elif particle.e_num == 20:
                                particle.i20 = particle.current_reef_idx
                                particle.ts20 = particle.current_reef_ts0
                                particle.dt20 = particle.current_reef_ts
                            elif particle.e_num == 21:
                                particle.i21 = particle.current_reef_idx
                                particle.ts21 = particle.current_reef_ts0
                                particle.dt21 = particle.current_reef_ts
                            elif particle.e_num == 22:
                                particle.i22 = particle.current_reef_idx
                                particle.ts22 = particle.current_reef_ts0
                                particle.dt22 = particle.current_reef_ts
                            elif particle.e_num == 23:
                                particle.i23 = particle.current_reef_idx
                                particle.ts23 = particle.current_reef_ts0
                                particle.dt23 = particle.current_reef_ts
                            elif particle.e_num == 24:
                                particle.i24 = particle.current_reef_idx
                                particle.ts24 = particle.current_reef_ts0
                                particle.dt24 = particle.current_reef_ts
                            elif particle.e_num == 25:
                                particle.i25 = particle.current_reef_idx
                                particle.ts25 = particle.current_reef_ts0
                                particle.dt25 = particle.current_reef_ts
                            elif particle.e_num == 26:
                                particle.i26 = particle.current_reef_idx
                                particle.ts26 = particle.current_reef_ts0
                                particle.dt26 = particle.current_reef_ts
                            elif particle.e_num == 27:
                                particle.i27 = particle.current_reef_idx
                                particle.ts27 = particle.current_reef_ts0
                                particle.dt27 = particle.current_reef_ts
                            elif particle.e_num == 28:
                                particle.i28 = particle.current_reef_idx
                                particle.ts28 = particle.current_reef_ts0
                                particle.dt28 = particle.current_reef_ts
                            elif particle.e_num == 29:
                                particle.i29 = particle.current_reef_idx
                                particle.ts29 = particle.current_reef_ts0
                                particle.dt29 = particle.current_reef_ts
                            elif particle.e_num == 30:
                                particle.i30 = particle.current_reef_idx
                                particle.ts30 = particle.current_reef_ts0
                                particle.dt30 = particle.current_reef_ts
                            elif particle.e_num == 31:
                                particle.i31 = particle.current_reef_idx
                                particle.ts31 = particle.current_reef_ts0
                                particle.dt31 = particle.current_reef_ts
                            elif particle.e_num == 32:
                                particle.i32 = particle.current_reef_idx
                                particle.ts32 = particle.current_reef_ts0
                                particle.dt32 = particle.current_reef_ts
                            elif particle.e_num == 33:
                                particle.i33 = particle.current_reef_idx
                                particle.ts33 = particle.current_reef_ts0
                                particle.dt33 = particle.current_reef_ts
                            elif particle.e_num == 34:
                                particle.i34 = particle.current_reef_idx
                                particle.ts34 = particle.current_reef_ts0
                                particle.dt34 = particle.current_reef_ts
                            elif particle.e_num == 35:
                                particle.i35 = particle.current_reef_idx
                                particle.ts35 = particle.current_reef_ts0
                                particle.dt35 = particle.current_reef_ts
                            elif particle.e_num == 36:
                                particle.i36 = particle.current_reef_idx
                                particle.ts36 = particle.current_reef_ts0
                                particle.dt36 = particle.current_reef_ts
                            elif particle.e_num == 37:
                                particle.i37 = particle.current_reef_idx
                                particle.ts37 = particle.current_reef_ts0
                                particle.dt37 = particle.current_reef_ts
                            elif particle.e_num == 38:
                                particle.i38 = particle.current_reef_idx
                                particle.ts38 = particle.current_reef_ts0
                                particle.dt38 = particle.current_reef_ts
                            elif particle.e_num == 39:
                                particle.i39 = particle.current_reef_idx
                                particle.ts39 = particle.current_reef_ts0
                                particle.dt39 = particle.current_reef_ts
                            elif particle.e_num == 40:
                                particle.i40 = particle.current_reef_idx
                                particle.ts40 = particle.current_reef_ts0
                                particle.dt40 = particle.current_reef_ts
                            elif particle.e_num == 41:
                                particle.i41 = particle.current_reef_idx
                                particle.ts41 = particle.current_reef_ts0
                                particle.dt41 = particle.current_reef_ts
                            elif particle.e_num == 42:
                                particle.i42 = particle.current_reef_idx
                                particle.ts42 = particle.current_reef_ts0
                                particle.dt42 = particle.current_reef_ts
                            elif particle.e_num == 43:
                                particle.i43 = particle.current_reef_idx
                                particle.ts43 = particle.current_reef_ts0
                                particle.dt43 = particle.current_reef_ts
                            elif particle.e_num == 44:
                                particle.i44 = particle.current_reef_idx
                                particle.ts44 = particle.current_reef_ts0
                                particle.dt44 = particle.current_reef_ts
                            elif particle.e_num == 45:
                                particle.i45 = particle.current_reef_idx
                                particle.ts45 = particle.current_reef_ts0
                                particle.dt45 = particle.current_reef_ts
                            elif particle.e_num == 46:
                                particle.i46 = particle.current_reef_idx
                                particle.ts46 = particle.current_reef_ts0
                                particle.dt46 = particle.current_reef_ts
                            elif particle.e_num == 47:
                                particle.i47 = particle.current_reef_idx
                                particle.ts47 = particle.current_reef_ts0
                                particle.dt47 = particle.current_reef_ts
                            elif particle.e_num == 48:
                                particle.i48 = particle.current_reef_idx
                                particle.ts48 = particle.current_reef_ts0
                                particle.dt48 = particle.current_reef_ts
                            elif particle.e_num == 49:
                                particle.i49 = particle.current_reef_idx
                                particle.ts49 = particle.current_reef_ts0
                                particle.dt49 = particle.current_reef_ts
                            elif particle.e_num == 50:
                                particle.i50 = particle.current_reef_idx
                                particle.ts50 = particle.current_reef_ts0
                                particle.dt50 = particle.current_reef_ts
                            elif particle.e_num == 51:
                                particle.i51 = particle.current_reef_idx
                                particle.ts51 = particle.current_reef_ts0
                                particle.dt51 = particle.current_reef_ts
                            elif particle.e_num == 52:
                                particle.i52 = particle.current_reef_idx
                                particle.ts52 = particle.current_reef_ts0
                                particle.dt52 = particle.current_reef_ts
                            elif particle.e_num == 53:
                                particle.i53 = particle.current_reef_idx
                                particle.ts53 = particle.current_reef_ts0
                                particle.dt53 = particle.current_reef_ts
                            elif particle.e_num == 54:
                                particle.i54 = particle.current_reef_idx
                                particle.ts54 = particle.current_reef_ts0
                                particle.dt54 = particle.current_reef_ts
                            elif particle.e_num == 55:
                                particle.i55 = particle.current_reef_idx
                                particle.ts55 = particle.current_reef_ts0
                                particle.dt55 = particle.current_reef_ts
                            elif particle.e_num == 56:
                                particle.i56 = particle.current_reef_idx
                                particle.ts56 = particle.current_reef_ts0
                                particle.dt56 = particle.current_reef_ts
                            elif particle.e_num == 57:
                                particle.i57 = particle.current_reef_idx
                                particle.ts57 = particle.current_reef_ts0
                                particle.dt57 = particle.current_reef_ts
                            elif particle.e_num == 58:
                                particle.i58 = particle.current_reef_idx
                                particle.ts58 = particle.current_reef_ts0
                                particle.dt58 = particle.current_reef_ts
                            elif particle.e_num == 59:
                                particle.i59 = particle.current_reef_idx
                                particle.ts59 = particle.current_reef_ts0
                                particle.dt59 = particle.current_reef_ts

                                particle.active = 0 # Deactivate particle, since no more reefs can be saved

                            # Then reset current values to zero
                            particle.current_reef_idx = 0
                            particle.current_reef_ts0 = 0
                            particle.current_reef_ts = 0

                            # Add to event number counter
                            particle.e_num += 1

                        if new_event:
                            # Add status to current (for current event) values
                            # Timesteps at current reef
                            particle.current_reef_ts = 1

                            # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                            particle.current_reef_ts0 = particle.ot - 1

                            # Current reef group
                            particle.current_reef_idx = particle.idx

                    # Finally, check if particle needs to be deleted
                    if particle.ot >= fieldset.max_age:

                        # Only delete particles where at least 1 event has been recorded
                        if particle.e_num > 0:
                            particle.delete()

        return event

    def run(self, **kwargs):
        """
        Run the configured OceanParcels simulation

        """

        if not self.status['particleset']:
            raise Exception('Please run particleset first.')

        self.fh['traj'] = self.dirs['traj'] + self.name + '.nc'

        print('Exporting output to ' + str(self.fh['traj']))

        if self.cfg['test']:
            if self.cfg['test_type'] == 'traj':
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(hours=0.25))
            elif self.cfg['test_type'] == 'vis':
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(hours=0.5))
            elif self.cfg['test_type'] == 'sens':
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(days=10))
            else:
                self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], write_ondelete=True)
        else:
            self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], write_ondelete=True)

        def deleteParticle(particle, fieldset, time):
            #  Recovery kernel to delete a particle if an error occurs
            particle.delete()

        def deactivateParticle(particle, fieldset, time):
            # Recovery kernel to deactivate a particle if an OOB error occurs
            particle.active = 0
            particle.lon = 40
            particle.lat = -1

        # Print some basic statistics
        print('')
        print('Starting simulation:')
        print('Name: ' + self.name)
        print('Number of release cells: ' + str(self.cfg['nsite']) + '/' + str(self.cfg['nsite_nofilter']))
        print('Number of particles released: ' + str(len(self.particles['lon'])))
        print('Release time: ' + str(self.particles['t0'].iloc[0]))
        print('Simulation length: ' + str(self.cfg['run_time']))

        if self.cfg['partitions']:
            print('Partition: ' + str(self.cfg['part']) + '/' + str(self.cfg['partitions']))

        print('')

        # Run the simulation
        self.pset.execute(self.kernel,
                          runtime=self.cfg['run_time'],
                          dt=self.cfg['dt'],
                          recovery={ErrorCode.ErrorOutOfBounds: deactivateParticle,
                                    ErrorCode.ErrorInterpolation: deleteParticle},
                          output_file=self.trajectory_file)

        # Export trajectory file
        self.trajectory_file.export()

        # Add timestep and other details to file
        with Dataset(self.fh['traj'], mode='r+') as nc:
            nc.timestep_seconds = self.cfg['dt'].total_seconds()
            nc.min_competency_seconds = self.cfg['min_competency'].total_seconds()
            nc.max_lifespan_seconds = self.cfg['run_time'].total_seconds()
            nc.larvae_per_cell = self.cfg['pn2']
            nc.total_larvae_released = len(self.particles['lon'])
            nc.interp_method = self.cfg['interp_method']
            nc.e_num = self.cfg['e_num']
            nc.release_year = self.cfg['t0'].year
            nc.release_month = self.cfg['t0'].month
            nc.release_day = self.cfg['t0'].day

            if self.cfg['test']:
                nc.test_mode = 'True'

            nc.partitions = int(self.cfg['partitions'])
            if self.cfg['partitions']:
                nc.part = int(self.cfg['part'])

        self.status['run'] = True


    def generate_dict(self):
        """
        Generate a dict to convert cell indices to reef cover, reef fraction, etc.

        """

        if not self.status['config']:
            raise Exception('Please run config first.')

        if not self.status['fieldset']:
            # Load fields
            self.fh['grid'] = self.dirs['grid'] + self.cfg['grid_filename']
            self.fields = {}

            self.field_list = ['rc', 'rf', 'eez', 'grp', 'idx']

            with Dataset(self.fh['grid'], mode='r') as nc:
                for field in self.field_list:
                    field_varname = self.cfg['grid_' + field + '_varname']
                    self.fields[field] = nc.variables[field_varname][:]

        self.dicts = {}

        # Firstly generate list of indices
        index_list = []
        for (yidx, xidx) in zip(np.ma.nonzero(self.fields['idx'])[0],
                                np.ma.nonzero(self.fields['idx'])[1]):

            index_list.append(self.fields['idx'][yidx, xidx])

        # Now generate dictionaries
        for field in self.field_list:
            if field != 'idx':
                temp_list = []

                for (yidx, xidx) in zip(np.ma.nonzero(self.fields['idx'])[0],
                                        np.ma.nonzero(self.fields['idx'])[1]):

                    temp_list.append(self.fields[field][yidx, xidx])
                    self.dicts[field] = dict(zip(index_list, temp_list))
                    self.dicts[field][0] = -999


        # Create dictionary to translate group -> number of cells in group
        grp_key, grp_val = np.unique(self.fields['grp'].compressed(),return_counts=True)
        self.dicts['grp_numcell'] = dict(zip(grp_key, grp_val))

        self.status['dict'] = True


    def process(self, **kwargs):
        """
        Generate matrices from SECoW output.

        Parameters
        ----------
        fh: List of file handles to data (not used if in testing mode)
        parameters: Processing parameters (in dict)

        testing: Boolean switch, turn on testing mode for validation
        subset: Int, take every [subset] trajectories to accelerate validation
        dt: timedelta, time-step for validation integration


        Returns
        -------
        Dictionary of matrices (based on options chosen in kwargs)

        """

        #######################################################################
        # PREPROCESSING OF PARAMETERS, CHECKS, MODES ##########################
        #######################################################################

        print('##########################################################')
        print('Preprocessing...')

        # Define conversion factor (seconds per year)
        conv_f = 31536000
        conv_day = conv_f/86400.

        # Cumulative day total for a year
        day_mo = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        day_mo_cs = np.cumsum(day_mo)
        day_mo_ly = np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        day_mo_ly_cs = np.cumsum(day_mo_ly)

        # Generate dict object
        if not self.status['dict']:
            self.generate_dict()

        # Biological larval parameters and unit conversion
        if 'parameters' not in kwargs.keys():
            raise KeyError('Please supply a parameters dictionary.')
        else:
            # Convert all units to years to prevent overflows from large numbers
            self.cfg['a'] = np.array(kwargs['parameters']['a']*conv_f, dtype=np.float64)
            self.cfg['b'] = np.array(kwargs['parameters']['b']*conv_f, dtype=np.float64)
            self.cfg['tc'] = np.array(kwargs['parameters']['tc']/conv_f, dtype=np.float64)
            self.cfg['μs'] = np.array(kwargs['parameters']['μs']*conv_f, dtype=np.float64)
            self.cfg['σ'] = np.array(kwargs['parameters']['σ'], dtype=np.float64)
            self.cfg['λ'] = np.array(kwargs['parameters']['λ']*conv_f, dtype=np.float64)
            self.cfg['ν'] = np.array(kwargs['parameters']['ν'], dtype=np.float64)

        # Test mode settings and defaults
        test_params = {}
        if 'test' in kwargs:
            if kwargs['test']:
                testing = True

                if 'subset' in kwargs:
                    test_params['subset'] = int(kwargs['subset'])
                else:
                    test_params['subset'] = int(1)

                if 'full_test' in kwargs:
                    full_test = kwargs['full_test']

                    if 'test_dt' in kwargs:
                        test_params['full_int_dt'] = kwargs['test_dt'] # (in seconds)
                    else:
                        test_params['full_int_dt'] = 600.
                else:
                    full_test = False
            else:
                testing = False
                full_test = False
                test_params['subset'] = int(1)
        else:
            testing = False
            full_test = False
            test_params['subset'] = int(1)

        # Get number of RK4 iterations
        if 'RK4_its' in kwargs:
            RK4_its = kwargs['RK4_its']
        else:
            RK4_its = 1

        # Get file handles
        if 'fh' not in kwargs.keys():
            raise KeyError('Please supply a list of files to analyse.')
        else:
            fh_list = sorted(glob(self.dirs['traj'] + kwargs['fh']))
            if testing:
                fh_list = [fh_list[0]]

        # Export modes
        export_mode = {}

        def set_export_mode(mode, export_mode):
            if mode in kwargs:
                export_mode[mode] = True if kwargs[mode] else False
            else:
                export_mode[mode] = False

            return export_mode

        mode_list = ['flux_src_grp', 'drift_time_grp', 'src_str_cell', 'snk_str_cell',
                     'ret_str_cell', 'src_ent_cell', 'snk_ent_cell', 'flux_src_cell',
                     'src_tst_cell', 'snk_tst_cell', 'src_cru_cell', 'snk_cru_cell']

        for mode in mode_list:
            export_mode = set_export_mode(mode, export_mode)

        if 'include_cells' in kwargs:
            # Filter kwarg should be equal to a list of cell indices that will
            # be used. Only cells starting from and arriving at the specified
            # cell indices will be exported.

            # Note that filter mode currently only works with the flux matrix,
            # and overrides other settings.

            filter_on = True
            include_cells = kwargs['include_cells']

            for mode in mode_list:
                export_mode[mode] = False
            export_mode['flux_src_grp'] = True

            print('Filtered mode activated. Exporting flux matrix only.')
            print('')

        else:
            filter_on = False

        if np.any([export_mode[mode] for mode in mode_list if 'cell' in mode]):
            export_mode['cell'] = True
        else:
            export_mode['cell'] = False

        if np.any([export_mode[mode] for mode in mode_list if 'grp' in mode]):
            export_mode['grp'] = True
        else:
            export_mode['grp'] = False

        if not np.any([export_mode[mode] for mode in mode_list]):
            print('Warning: no export mode selected!')
            print('Available export modes:')
            print('flux_src_grp, snk_entr_cell, drift_time_grp, src_str_cell')
            print('snk_str_cell, ret_str_cell, src_ent_cell, flux_src_cell')
            return None

        # Define translation function (pass an array through a dictionary)
        def translate(c1, c2):
            # c1: Indices (i.e. array of keys to translate to values)
            # c2: Dictionary (keys: values)
            src, values = np.array(list(c2.keys()), dtype=np.uint16), np.array(list(c2.values()), dtype=np.float32)
            c2_array = np.zeros((src.max()+1), dtype=np.float32)
            c2_array[src] = values
            return c2_array[c1]

        # Open the first file to find the number of events stored and remaining parameters
        with xr.open_dataset(fh_list[0]) as file:
            self.cfg['max_events'] = file.attrs['e_num']

            self.cfg['dt'] = int(file.attrs['timestep_seconds'])/conv_f
            self.cfg['lpc'] = int(file.attrs['larvae_per_cell'])

            if testing:
                _lpc_safe_threshold = (self.cfg['lpc']**0.5)/2
                self.cfg['lpc'] = int(np.ceil(self.cfg['lpc']/test_params['subset']))
                if self.cfg['lpc'] < _lpc_safe_threshold:
                    print('Warning: subsetting is high - may be poorly sampling cells.')

                y0 = int(file.attrs['release_year'])

            if self.cfg['tc']*conv_f < int(file.attrs['min_competency_seconds']):
                print('Warning: minimum competency chosen is smaller than the value used at run-time (' + str(int(file.attrs['min_competency_seconds'])) +'s).')
                print('Setting minimum competency to model minimum (' + str(int(file.attrs['min_competency_seconds'])) +'s).')
                self.cfg['tc'] = int(file.attrs['min_competency_seconds'])/conv_f

        # Check all expected files are available (assuming grouping years together with 1 day = 1 traj file)
        if not testing:
            t0_list = []

            for fh in fh_list:
                with xr.open_dataset(fh) as file:
                    t0_list.append(int(file.attrs['release_year']))

            t0_list = np.array(t0_list)

            # Check all files are present
            if len(np.unique(t0_list)) != 1:
                raise Exception('More than year present!')
            else:
                y0 = t0_list[0]

            ly = False if y0%4 else True
            n_days = 366 if ly else 365

            if n_days != len(fh_list):
                raise Exception('Warning: there is an unexpected number of files!')

        else:
            ly = False if y0%4 else True
            n_days = 366 if ly else 365

        # Get a list of group and cell numbers
        reef_mask = (self.fields['rc'] > 0)

        cell_list = np.unique(np.ma.masked_where(reef_mask == 0, self.fields['idx'])).compressed()
        cell_bnds = np.append(cell_list, cell_list[-1]+1)-0.5
        cell_num = len(cell_list)

        if export_mode['grp']:
            if filter_on:
                grp_list = include_cells

                # Now override the group size filter
                self.dicts['grp_numcell'] = {sel_grp: 1 for sel_grp in grp_list}
            else:
                grp_list = np.unique(np.ma.masked_where(reef_mask == 0, self.fields['grp'])).compressed()

            grp_bnds = np.append(grp_list, grp_list[-1]+1)-0.5
            grp_num = len(grp_list)

            if filter_on:
                self.dicts['rc_grp'] = self.dicts['rc']
            else:
                self.dicts['rc_grp'] = {grp: 0. for grp in grp_list}
                for cell in cell_list:
                    self.dicts['rc_grp'][int(np.floor(cell/(2**8)))] += self.dicts['rc'][cell]

        # Set up matrices
        ns_mx = {}

        if export_mode['flux_src_grp']:
            ns_mx['flux_src_grp'] = np.zeros((grp_num, grp_num, n_days), dtype=np.float32) # Number settling

        if export_mode['drift_time_grp']:
            ns_mx['drift_time_grp'] = np.zeros((grp_num, grp_num, 240), dtype=np.float32) # Number settling
            drift_time_bnds = np.arange(0, 120.5, 0.5) # Days

        if export_mode['src_str_cell']:
            ns_mx['src_str_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Number settling

        if export_mode['snk_str_cell']:
            ns_mx['snk_str_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Number settling

        if export_mode['ret_str_cell']:
            ns_mx['ret_str_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Number settling

        if export_mode['src_ent_cell']:
            ns_mx['src_ent_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Source entropy

        if export_mode['snk_ent_cell']:
            ns_mx['snk_ent_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Sink entropy

        if export_mode['flux_src_cell']:
            ns_mx['flux_src_cell'] = np.zeros((cell_num, cell_num, 12), dtype=np.float32) # Number settling

        if export_mode['src_tst_cell']:
            ns_mx['src_tst_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Outbound TST

        if export_mode['snk_tst_cell']:
            ns_mx['snk_tst_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Inbound TST

        if export_mode['src_cru_cell']:
            ns_mx['src_cru_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Source CRU

        if export_mode['snk_cru_cell']:
            ns_mx['snk_cru_cell'] = np.zeros((cell_num, n_days), dtype=np.float32) # Sink CRU

        # Create attribute dictionary
        attr_dict = {}

        print('...done!')
        print('##########################################################')
        print('')

        #######################################################################
        # VALIDATION ##########################################################
        #######################################################################

        # This section only runs in testing mode. It independently produces
        # connectivity matrices by explicitly integrating the larval biology
        # equations.

        if full_test:
            print('##########################################################')
            print('Computing larval fluxes for validation...')

            ns_mx_full_int = {}

            if export_mode['flux_src_grp']:
                ns_mx_full_int['flux_src_grp'] = np.zeros((grp_num, grp_num, 1), dtype=np.float32)

            if export_mode['src_str_cell']:
                ns_mx_full_int['src_str_cell'] = np.zeros((cell_num, 1), dtype=np.float32)

            if export_mode['flux_src_cell']:
                ns_mx_full_int['flux_src_cell'] = np.zeros((cell_num, cell_num, 1), dtype=np.float32)

            with xr.open_dataset(fh_list[0], mask_and_scale=True) as file:
                # Get number of trajectories and time-steps
                full_int_dt = test_params['full_int_dt']
                refine = int(self.cfg['dt']*conv_f/full_int_dt)
                assert np.isclose(self.cfg['dt']*conv_f/full_int_dt, np.floor(self.cfg['dt']*conv_f/full_int_dt), 1e-9)

                _n_traj = len(file.e_num[::test_params['subset']]) # (!= 8088*1024 because not all virtual larvae record any events)
                _n_ts = int(file.max_lifespan_seconds/full_int_dt)

                # Create template matrices with index
                _idx_matrix = np.zeros((_n_traj, _n_ts), dtype=np.int32)
                _drift_time = np.zeros((_n_traj, self.cfg['max_events']), dtype=np.float32)

                # Get time-series for the mortality rate
                _time = np.linspace(full_int_dt/86400.,
                                    full_int_dt/86400. + (file.max_lifespan_seconds/86400),
                                    num=_n_ts + 1)

                _tc = self.cfg['tc']*conv_day
                _μs = self.cfg['μs']/conv_day
                _a = np.ones_like(_time, dtype=np.float32)*self.cfg['a']/conv_day
                _a[_time < _tc] = 0
                _b = np.ones_like(_time, dtype=np.float32)*self.cfg['b']/conv_day
                _b[_time < _tc] = 0
                _σ = self.cfg['σ']
                _λ = self.cfg['λ']/conv_day
                _ν = self.cfg['ν']
                _dt = full_int_dt/86400.

                if self.cfg['σ'] != 0:
                    _μm = (_λ*_ν*(_λ*_time)**(_ν-1))/(1-_σ*(_λ*_time)**_ν)
                else:
                    _μm = (_λ*_ν)*(_λ*_time)**(_ν-1)

                # Find the time-step at which competency begins
                # Issue is connected to starting time
                if _tc < file.attrs['min_competency_seconds']/86400.:
                    _ts_min = np.searchsorted(_time, (file.attrs['min_competency_seconds']/86400.))
                else:
                    _ts_min = np.searchsorted(_time, _tc)

                def populate_event(_matrix_idx, _n_traj, _ts_event, _dt_event, _i_event, _ts_min):
                    # Full events
                    _matrix_drift_time = (_ts_event + 0.5*_dt_event)*refine*full_int_dt
                    _matrix_drift_time = _matrix_drift_time*(_ts_event+1 >= (_ts_min+1)/refine)

                    # Partial events
                    partial_event = (_ts_event+_dt_event)*refine - _ts_min # Length of partial event
                    partial_event = partial_event.astype(int)
                    partial_event_bool = (partial_event > 0)*(_ts_event*refine < _ts_min)
                    _matrix_drift_time += ((_ts_min+1)/refine + 0.5*partial_event)*refine*full_int_dt*partial_event_bool

                    for _traj in range(_n_traj):
                        if _ts_event[_traj]*refine >= _ts_min:
                            _matrix_idx[_traj, _ts_event[_traj]*refine:(_ts_event[_traj]+_dt_event[_traj])*refine] = _i_event[_traj]
                        elif (_ts_event[_traj]+_dt_event[_traj])*refine >= _ts_min:
                            _matrix_idx[_traj, _ts_min:_ts_min+partial_event[_traj]] = _i_event[_traj]

                    return _matrix_idx, _matrix_drift_time

                _time_t0 = systime()

                for _event in range(self.cfg['max_events']):
                    _ts_event = file.variables['ts' + str(_event)][::test_params['subset']].values.astype(int)
                    _dt_event = file.variables['dt' + str(_event)][::test_params['subset']].values.astype(int)
                    _i_event = file.variables['i' + str(_event)][::test_params['subset']].values.astype(int)

                    _idx_matrix, _drift_time[:, _event] = populate_event(_idx_matrix, _n_traj, _ts_event, _dt_event, _i_event, _ts_min)

                # Now get corresponding matrix for reef fraction
                self.dicts['_rf'] = self.dicts['rf'].copy()
                self.dicts['_rf'][0] = 0
                _rf_matrix = translate(_idx_matrix, self.dicts['_rf'])

                # Now make output matrices
                idx_exp_full = np.zeros((_n_traj, self.cfg['max_events']), dtype=np.int32)
                ns_exp_full = np.zeros((_n_traj, self.cfg['max_events']), dtype=np.float32)

                @njit
                def validate_ns_rk4(_n_traj, _rf_matrix, _idx_matrix, _n_ts,
                                    _ts_min, _a, _b, _μm, _μs, _idx_out, _ns_out):
                    for _traj in range(_n_traj):
                        # Initial values for L1, L2
                        _L1 = 1.
                        _L2 = 0.
                        _rf_traj = _rf_matrix[_traj, :]
                        _idx_traj = _idx_matrix[_traj, :]
                        _current_idx = 0
                        _current_event = -1

                        for _ts in range(_n_ts):
                            _i = _idx_traj[_ts]

                            # Use one RK4 step
                            if _i != 0:
                                _rf = _rf_traj[_ts]

                                # kx1 = L1 // kx2 = L2 // kx3 = ns

                                k11 = -(_a[_ts] + _μm[_ts])*_L1
                                k12 = (_a[_ts]*_L1) - (_b[_ts] + _μm[_ts] + (_μs*_rf))*_L2
                                k13 = _μs*_rf*_L2

                                k21 = -(_a[_ts] + _μm[_ts])*(_L1+(_dt*0.5*k11))
                                k22 = (_a[_ts]*(_L1+(_dt*0.5*k11))) - (_b[_ts] + _μm[_ts] + (_μs*_rf))*(_L2+(_dt*0.5*k12))
                                k23 = _μs*_rf*(_L2+(_dt*0.5*k12))

                                k31 = -(_a[_ts] + _μm[_ts])*(_L1+(_dt*0.5*k21))
                                k32 = (_a[_ts]*(_L1+(_dt*0.5*k21))) - (_b[_ts] + _μm[_ts] + (_μs*_rf))*(_L2+(_dt*0.5*k22))
                                k33 = _μs*_rf*(_L2+(_dt*0.5*k22))

                                k41 = -(_a[_ts] + _μm[_ts+1])*(_L1+(_dt*k31))
                                k42 = (_a[_ts]*(_L1+(_dt*k31))) - (_b[_ts] + _μm[_ts+1] + (_μs*_rf))*(_L2+(_dt*k32))
                                k43 = _μs*_rf*(_L2+(_dt*k32))

                                _dL1 = (k11 + 2*k21 + 2*k31 + k41)*_dt/6
                                _dL2 = (k12 + 2*k22 + 2*k32 + k42)*_dt/6
                                _dns = (k13 + 2*k23 + 2*k33 + k43)*_dt/6

                                if _i != _current_idx:
                                    _current_event += 1
                                    _current_idx = _i
                                    _idx_out[_traj, _current_event] = _i

                                _ns_out[_traj, _current_event] += _dns

                            else:
                                # kx1 = L1 // kx2 = L2 // kx3 = ns

                                k11 = -(_a[_ts] + _μm[_ts])*_L1
                                k12 = (_a[_ts]*_L1) - (_b[_ts] + _μm[_ts])*_L2

                                k21 = -(_a[_ts] + _μm[_ts])*(_L1+(_dt*0.5*k11))
                                k22 = (_a[_ts]*(_L1+(_dt*0.5*k11))) - (_b[_ts] + _μm[_ts])*(_L2+(_dt*0.5*k12))

                                k31 = -(_a[_ts] + _μm[_ts])*(_L1+(_dt*0.5*k21))
                                k32 = (_a[_ts]*(_L1+(_dt*0.5*k21))) - (_b[_ts] + _μm[_ts])*(_L2+(_dt*0.5*k22))

                                k41 = -(_a[_ts] + _μm[_ts+1])*(_L1+(_dt*k31))
                                k42 = (_a[_ts]*(_L1+(_dt*k31))) - (_b[_ts] + _μm[_ts+1])*(_L2+(_dt*k32))

                                _dL1 = (k11 + 2*k21 + 2*k31 + k41)*_dt/6
                                _dL2 = (k12 + 2*k22 + 2*k32 + k42)*_dt/6

                                if _i != _current_idx:
                                    _current_idx = _i

                            _L1 = _L1 + _dL1
                            _L2 = _L2 + _dL2

                    return _idx_out, _ns_out

                idx_exp_full, ns_exp_full = validate_ns_rk4(_n_traj, _rf_matrix, _idx_matrix,
                                                            _n_ts, _ts_min, _a, _b, _μm, _μs,
                                                            idx_exp_full, ns_exp_full)

                print('Full explicit integration time: ')
                print(str(round(systime() - _time_t0, 1)) + 's')
                ns_array_full_int_testing = ns_exp_full.copy()

                # Compress earlier to accelerate calculations
                # Convert results into a 1D array
                _mask = idx_exp_full == 0
                _mask_shape = np.shape(_mask)

                ns_exp_full = np.ma.masked_array(ns_exp_full, mask=_mask).compressed()
                _snk_cell = np.ma.masked_array(idx_exp_full, mask=_mask).compressed()
                _src_cell = np.zeros(_mask_shape, dtype=int)
                _src_cell[:] = file.idx0[::test_params['subset']].values.reshape((-1,1))
                _src_cell = np.ma.masked_array(_src_cell, mask=_mask).compressed()

                # Extract origin reef cover
                _src_rc = translate(_src_cell, self.dicts['rc'])

                # Convert cell dtypes to float (to avoid causing problems with binning)
                _src_cell = _src_cell.astype(np.float32)
                _snk_cell = _snk_cell.astype(np.float32)

                # Compute flux
                flux = ns_exp_full*_src_rc/self.cfg['lpc']

                # Compute groups if necessary
                if np.any(['grp' in _mode for _mode in export_mode]):
                    _src_grp = np.floor(_src_cell/(2**8)).astype(np.float32)
                    _snk_grp = np.floor(_snk_cell/(2**8)).astype(np.float32)

                if export_mode['flux_src_grp']:
                    ns_mx_full_int['flux_src_grp'][:, :, 0] += np.histogram2d(_src_grp, _snk_grp,
                                                                          bins=[grp_bnds, grp_bnds],
                                                                          weights=flux)[0]

                if export_mode['src_str_cell']:
                    ns_mx_full_int['src_str_cell'][:, 0] += np.histogramdd((_src_cell), bins=[cell_bnds], weights=flux)[0]

                if export_mode['flux_src_cell']:
                    ns_mx_full_int['flux_src_cell'][:, :, 0] += np.histogram2d(_src_cell, _snk_cell,
                                                                           bins=[cell_bnds, cell_bnds],
                                                                           weights=flux)[0]
            print('...done!')
            print('##########################################################')
            print('')

        #######################################################################
        # MAIN ROUTINE #########################################################
        #######################################################################

        print('##########################################################')
        print('Computing larval fluxes...')

        for fhi, fh in tqdm(enumerate(fh_list), total=len(fh_list)):
            with xr.open_dataset(fh, mask_and_scale=False) as file:

                n_traj = len(file.e_num[::test_params['subset']])

                if not n_traj:
                    # Skip if there are no trajectories stored in file
                    raise Exception('No trajectories found in file ' + str(fh) + '!')

                # Extract origin date from filename
                y0 = int(file.attrs['release_year'])
                m0 = int(file.attrs['release_month'])
                d0 = int(file.attrs['release_day'])
                assert y0 < 2025 and y0 > 1990
                assert m0 < 13 and m0 > 0
                assert d0 < 32 and d0 > 0

                # Load all data into memory
                idx_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.uint16)
                t0_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)
                dt_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)
                ns_array = np.zeros((n_traj, self.cfg['max_events']), dtype=np.float32)

                for i in range(self.cfg['max_events']):
                    if testing:
                        idx_array[:, i] = file['i' + str(i)][::test_params['subset']].values
                        t0_array[:, i] = (file['ts' + str(i)][::test_params['subset']].values*self.cfg['dt'])-self.cfg['tc'] # Time at arrival
                        dt_array[:, i] = file['dt' + str(i)][::test_params['subset']].values*self.cfg['dt'] # Time spent at site
                    else:
                        idx_array[:, i] = file['i' + str(i)].values
                        t0_array[:, i] = (file['ts' + str(i)].values*self.cfg['dt'])-self.cfg['tc'] # Time at arrival
                        dt_array[:, i] = file['dt' + str(i)].values*self.cfg['dt'] # Time spent at site

                if testing:
                    idx0_array = file['idx0'][::test_params['subset']].values.astype(np.uint16)
                else:
                    idx0_array = file['idx0'].values.astype(np.uint16)

                # Implement filter if required
                if filter_on:
                    idx0_filter = np.isin(idx0_array, include_cells)
                    idx_array = idx_array[idx0_filter, :]
                    t0_array = t0_array[idx0_filter, :]
                    dt_array = dt_array[idx0_filter, :]
                    idx0_array = idx0_array[idx0_filter]
                    n_traj = len(idx0_array)

                # Adjust times for events that are partially pre-competent
                idx_array[t0_array + dt_array < 0] = 0 # Corresponds to events that are entirely precompetent - ignore (via 0 in idx_array, see below)
                dt_array[t0_array < 0] += t0_array[t0_array < 0] # For partially precompetent events, compute the post-competent time
                t0_array[t0_array < 0] = 0 # Set new starting time to zero for partially precompetent events

                mask = (idx_array == 0) # Mask out entirely precompetent events

                ns_array = np.zeros(np.shape(mask), dtype=np.float32) # Output is f32

                # Copy over parameters for output file
                for attr_name in ['parcels_version', 'timestep_seconds',
                                  'min_competency_seconds', 'max_lifespan_seconds',
                                  'larvae_per_cell', 'interp_method', 'e_num',
                                  'release_year']:
                    if fhi == 0:
                        attr_dict[attr_name] = file.attrs[attr_name]
                    else:
                        assert attr_dict[attr_name] == file.attrs[attr_name]

                if fhi == 0:
                    attr_dict['total_larvae_released'] = file.attrs['total_larvae_released']
                    self.cfg['run_time'] = timedelta(seconds=int(attr_dict['max_lifespan_seconds']))
                else:
                    attr_dict['total_larvae_released'] += file.attrs['total_larvae_released']

                # Now generate an array containing the reef fraction, t0, and dt for each index
                rf_array = translate(idx_array, self.dicts['rf'])

                # Set fr/t0/dt to 0 for all invalid events (to prevent pre-competent events
                # from ruining integrals)
                rf_array[mask] = 0 # Reef fraction
                t0_array[mask] = 0 # Time at arrival
                dt_array[mask] = 0 # Time spent at site

                if testing:
                    _time_t0 = systime()

                for i in range(self.cfg['max_events']):
                    if i == 0:
                        psi0 = np.zeros((n_traj,), dtype=np.float64)
                        int0 = np.zeros((n_traj,), dtype=np.float64)
                        t1_prev = np.zeros((n_traj,), dtype=np.float64)

                    rf = rf_array[:, i].astype(np.float64)
                    t0 = t0_array[:, i].astype(np.float64)
                    dt = dt_array[:, i].astype(np.float64)

                    # Use analytic scheme for analytic case, otherwise semi-analytic
                    ns_array[:, i], int0, psi0 = self.integrate(psi0, int0, rf,
                                                                self.cfg['a'],
                                                                self.cfg['b'],
                                                                self.cfg['tc'],
                                                                self.cfg['μs'],
                                                                self.cfg['σ'],
                                                                self.cfg['λ'],
                                                                self.cfg['ν'],
                                                                t0, t1_prev, dt,
                                                                RK4_its)

                    t1_prev = t0 + dt

                if testing:
                    print('Standard integration time: ')
                    print(str(round(systime() - _time_t0, 1)) + 's')

                    if fhi == 0:
                        ns_array_testing = ns_array.copy()

                # Compress earlier to accelerate calculations
                ns_array = np.ma.masked_array(ns_array, mask=mask).compressed()
                t0_array = np.ma.masked_array(t0_array, mask=mask).compressed()
                dt_array = np.ma.masked_array(dt_array, mask=mask).compressed()

                assert np.all(ns_array >= 0)
                assert np.all(t0_array >= 0)
                assert np.all(dt_array >= 0)

                # From the cell array, extract group
                if export_mode['grp']:
                    if filter_on:
                        snk_grp = np.ma.masked_array(idx_array, mask=mask).compressed()

                        src_grp = np.zeros_like(idx_array, dtype=np.uint16)
                        src_grp[:] = idx0_array.reshape((-1, 1))
                        src_grp = np.ma.masked_array(src_grp, mask=mask).compressed()

                        # Now implement filter
                        idx_filter = np.isin(snk_grp, include_cells)
                        ns_array = ns_array[idx_filter]
                        t0_array = t0_array[idx_filter]
                        dt_array = dt_array[idx_filter]
                        src_grp = src_grp[idx_filter]
                        snk_grp = snk_grp[idx_filter]

                    else:
                        snk_grp = np.floor(idx_array/(2**8)).astype(np.uint8)
                        snk_grp = np.ma.masked_array(snk_grp, mask=mask).compressed()

                        src_grp_ = np.floor(idx0_array/(2**8)).astype(np.uint8)
                        src_grp = np.zeros_like(idx_array, dtype=np.uint8)
                        src_grp[:] = src_grp_.reshape((-1, 1))
                        src_grp = np.ma.masked_array(src_grp, mask=mask).compressed()

                # Also extract cells
                if export_mode['cell']:
                    snk_cell = np.ma.masked_array(idx_array, mask=mask).compressed()

                    src_cell = np.zeros_like(idx_array, dtype=np.uint16)
                    src_cell[:] = idx0_array.reshape((-1, 1))
                    src_cell = np.ma.masked_array(src_cell, mask=mask).compressed()

                # Extract origin reef cover
                src_rc = np.zeros_like(idx_array, dtype=np.float32)
                src_rc[:] = translate(idx0_array, self.dicts['rc']).reshape((-1, 1))
                src_rc = np.ma.masked_array(src_rc, mask=mask).compressed()

                if filter_on:
                    src_rc = src_rc[idx_filter]

                # Compute drift time (in days)
                drift_time = (self.cfg['tc'] + t0_array + 0.5*dt_array)*conv_f/86400.

                # Compute retention mask if applicable
                if export_mode['ret_str_cell']:
                    ret_mask = (src_cell == snk_cell).astype(int)

                # Find time index
                if ly:
                    ti = day_mo_ly_cs[m0-1] + d0 - 1
                else:
                    ti = day_mo_cs[m0-1] + d0 - 1

                # Compute the actual larval flux
                flux = ns_array*src_rc/self.cfg['lpc']

                ###############################################################
                # GRID FLUXES                                                 #
                ###############################################################

                # NOTE:
                # The sink routines are currently very memory intensive, since they allow sinking to occur at
                # any time. In practice, a sink event can only occur within 120 days after spawning, so
                # if you find that you are memory limited, there are some relatively easy gains to be
                # made here.

                if np.array([export_mode['src_str_cell'] + export_mode['snk_str_cell'] +
                             export_mode['ret_str_cell'] + export_mode['src_tst_cell'] +
                             export_mode['snk_tst_cell'] + export_mode['src_cru_cell'] +
                             export_mode['snk_cru_cell'] + export_mode['src_ent_cell'] +
                             export_mode['snk_ent_cell'] + export_mode['flux_src_cell']]).any():
                    cell_flux_base = np.histogram2d(src_cell, snk_cell, bins=[cell_bnds, cell_bnds], weights=flux)[0]

                if export_mode['flux_src_grp']:
                    ns_mx['flux_src_grp'][:, :, ti] = np.histogram2d(src_grp, snk_grp,
                                                                     bins=[grp_bnds, grp_bnds],
                                                                     weights=flux)[0]

                if export_mode['drift_time_grp']:
                    ns_mx['drift_time_grp'] += np.histogramdd((src_grp, snk_grp, drift_time),
                                                              bins=[grp_bnds, grp_bnds, drift_time_bnds],
                                                              weights=flux)[0]

                if export_mode['src_str_cell']:
                    ns_mx['src_str_cell'][:, ti] = cell_flux_base.sum(axis=1)

                if export_mode['snk_str_cell']:
                    ns_mx['snk_str_cell'][:, ti] = cell_flux_base.sum(axis=0)

                if export_mode['ret_str_cell']:
                    ns_mx['ret_str_cell'][:, ti] = (cell_flux_base*np.identity(cell_num)).sum(axis=1)

                if export_mode['src_tst_cell']:
                    flux_tst = (np.copy(cell_flux_base)*(kwargs['tst'].reshape(1, -1))).sum(axis=1)
                    norm = cell_flux_base.sum(axis=1)
                    flux_tst[norm == 0] = np.nan # Undefined if there are no sinks
                    norm[norm == 0] = 1
                    ns_mx['src_tst_cell'][:, ti] = flux_tst/norm

                if export_mode['snk_tst_cell']:
                    flux_tst = (np.copy(cell_flux_base)*(kwargs['tst'].reshape(-1, 1))).sum(axis=0)
                    norm = cell_flux_base.sum(axis=0)
                    flux_tst[norm == 0] = np.nan # Undefined if there are no sources
                    norm[norm == 0] = 1
                    ns_mx['snk_tst_cell'][:, ti] = flux_tst/norm

                if export_mode['src_cru_cell']:
                    # i.e. the proportion of time the source reef IS NOT at bleaching risk that sink reef IS at bleaching risk
                    flux_cru = (np.copy(cell_flux_base)*(kwargs['src_cru'])).sum(axis=1)
                    norm = cell_flux_base.sum(axis=1)
                    flux_cru[norm == 0] = np.nan # Undefined if there are no sinks
                    norm[norm == 0] = 1
                    ns_mx['src_cru_cell'][:, ti] = flux_cru/norm

                if export_mode['snk_cru_cell']:
                    # i.e. the proportion of time the sink reef IS at bleaching risk that source reef IS NOT at bleaching risk
                    flux_cru = (np.copy(cell_flux_base)*(kwargs['snk_cru'])).sum(axis=0)
                    norm = cell_flux_base.sum(axis=0)
                    flux_cru[norm == 0] = np.nan # Undefined if there are no sources
                    norm[norm == 0] = 1
                    ns_mx['snk_cru_cell'][:, ti] = flux_cru/norm

                if export_mode['src_ent_cell']:
                    # This computation is more involved. We firstly need the full flux matrix for time ti
                    # (note 1 - this routine only works if there is 1 file per source day, which is true for
                    #  SECoW)
                    _flux = np.copy(cell_flux_base)
                    _flux_sum = np.sum(_flux, axis=1)
                    _flux_sum[_flux_sum == 0] = 1
                    _flux = _flux/(_flux_sum.reshape((len(_flux_sum), 1))) # Normalise by sink strength (we are not interested in dead larvae)

                    # Now compute the source entropy
                    _flux_mod = np.copy(_flux)
                    _flux_mod[_flux_mod == 0] = 1 # To avoid NaNs
                    ns_mx['src_ent_cell'][:, ti] = -np.sum(_flux*np.log2(_flux_mod), axis=1)

                if export_mode['snk_ent_cell']:
                    _flux = np.copy(cell_flux_base)
                    _flux_sum = np.sum(_flux, axis=0)
                    _flux_sum[_flux_sum == 0] = 1
                    _flux = _flux/(_flux_sum.reshape((1, len(_flux_sum)))) # Normalise by source strength (we are not interested in dead larvae)

                    # Now compute the sink entropy
                    _flux_mod = np.copy(_flux)
                    _flux_mod[_flux_mod == 0] = 1 # To avoid NaNs
                    ns_mx['snk_ent_cell'][:, ti] = -np.sum(_flux*np.log2(_flux_mod), axis=0)

                if export_mode['flux_src_cell']:
                    ns_mx['flux_src_cell'][:, :, m0-1] += np.copy(cell_flux_base)


            ###################################################################
            # EXPORT FLUXES                                                   #
            ###################################################################

            output = {}

            if export_mode['flux_src_grp']:
                output['flux_src_grp'] = xr.Dataset(data_vars=dict(ns=(['source_group', 'sink_group', 'time'], ns_mx['flux_src_grp'],
                                                                       {'full_name': 'number of settling larvae as a function of source time',
                                                                        'units': 'settling_larvae',}),
                                                                   cpg=(['source_group'], translate(grp_list, self.dicts['grp_numcell']),
                                                                        {'full_name': 'cells_per_group',
                                                                         'units': 'cells'}),
                                                                   rc=(['source_group'], translate(grp_list, self.dicts['rc_grp']),
                                                                       {'full_name': 'reef_cover',
                                                                       'units': 'm2'})),
                                                    coords=dict(source_group=grp_list, sink_group=grp_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['drift_time_grp']:
                output['drift_time_grp'] = xr.Dataset(data_vars=dict(ns=(['source_group', 'sink_group', 'drift_time'], ns_mx['drift_time_grp'],
                                                                         {'full_name': 'number of settling larvae as a function of drift time',
                                                                          'units': 'settling_larvae',}),
                                                                     cpg=(['source_group'], translate(grp_list, self.dicts['grp_numcell']),
                                                                          {'full_name': 'cells_per_group',
                                                                           'units': 'cells'}),
                                                                     rc=(['source_group'], translate(grp_list, self.dicts['rc_grp']),
                                                                         {'full_name': 'reef_cover',
                                                                         'units': 'm2'}),),
                                                      coords=dict(source_group=grp_list, sink_group=grp_list,
                                                                  drift_time=np.arange(0.25, 120.25, 0.5)),
                                                      attrs=dict(a=self.cfg['a']/conv_f,
                                                                 b=self.cfg['b']/conv_f,
                                                                 tc=self.cfg['tc']*conv_f,
                                                                 μs=self.cfg['μs']/conv_f,
                                                                 σ=self.cfg['σ'],
                                                                 λ=self.cfg['λ']/conv_f,
                                                                 ν=self.cfg['ν'],
                                                                 configuration=self.cfg['preset'],
                                                                 parcels_version=attr_dict['parcels_version'],
                                                                 timestep_seconds=attr_dict['timestep_seconds'],
                                                                 min_competency_seconds=attr_dict['min_competency_seconds'],
                                                                 max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                                 larvae_per_cell=attr_dict['larvae_per_cell'],
                                                                 interp_method=attr_dict['interp_method'],
                                                                 e_num=attr_dict['e_num'],
                                                                 total_larvae_released=attr_dict['total_larvae_released'],
                                                                 RK4_iterations=RK4_its))

            if export_mode['src_str_cell']:
                # i.e. the total number of larvae spawning from a cell that settle
                output['src_str_cell'] = xr.Dataset(data_vars=dict(ns=(['source_cell', 'time'], ns_mx['src_str_cell'],
                                                                       {'full_name': 'source strength for cells',
                                                                        'units': 'settling_larvae',}),
                                                                   rc=(['source_cell'], translate(cell_list, self.dicts['rc']),
                                                                        {'full_name': 'reef_cover',
                                                                         'units': 'm2'}),),
                                                    coords=dict(source_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['snk_str_cell']:
                # i.e the total number of larvae from a spawning event that settle in a cell
                output['snk_str_cell'] = xr.Dataset(data_vars=dict(ns=(['sink_cell', 'time'], ns_mx['snk_str_cell'],
                                                                       {'full_name': 'sink strength for cells',
                                                                        'units': 'settling_larvae',}),
                                                                   rc=(['sink_cell'], translate(cell_list, self.dicts['rc']),
                                                                       {'full_name': 'reef_cover',
                                                                        'units': 'm2'}),),
                                                    coords=dict(sink_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['ret_str_cell']:
                # i.e. the total number of larvae that settle in the same cell they spawned from
                output['ret_str_cell'] = xr.Dataset(data_vars=dict(ns=(['source_cell', 'time'], ns_mx['ret_str_cell'],
                                                                       {'full_name': 'retention strength for cells',
                                                                        'units': 'settling_larvae',}),
                                                                   rc=(['source_cell'], translate(cell_list, self.dicts['rc']),
                                                                       {'full_name': 'reef_cover',
                                                                        'units': 'm2'})),
                                                    coords=dict(source_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['src_ent_cell']:
                # i.e. the entropy of larval destinations from a source cell
                output['src_ent_cell'] = xr.Dataset(data_vars=dict(entropy=(['source_cell', 'time'], ns_mx['src_ent_cell'],
                                                                            {'full_name': 'cell source entropy',
                                                                             'units': 'bits',}),
                                                                   rc=(['source_cell'], translate(cell_list, self.dicts['rc']),
                                                                        {'full_name': 'reef_cover',
                                                                         'units': 'm2'})),
                                                    coords=dict(source_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['snk_ent_cell']:
                # i.e. the entropy of larval sources for a sink cell
                output['snk_ent_cell'] = xr.Dataset(data_vars=dict(entropy=(['sink_cell', 'time'], ns_mx['snk_ent_cell'],
                                                                            {'full_name': 'cell sink entropy',
                                                                             'units': 'bits',}),
                                                                   rc=(['sink_cell'], translate(cell_list, self.dicts['rc']),
                                                                        {'full_name': 'reef_cover',
                                                                         'units': 'm2'})),
                                                    coords=dict(sink_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['src_tst_cell']:
                # i.e. the mean TST for downstream cells from a source cell
                output['src_tst_cell'] = xr.Dataset(data_vars=dict(downstream_tst=(['source_cell', 'time'], ns_mx['src_tst_cell'],
                                                                                   {'full_name': 'mean downstream TST',
                                                                                    'units': 'deg C',}),
                                                                   tst=(['source_cell'], kwargs['tst'],
                                                                        {'full_name': 'source cell TST',
                                                                         'units': 'deg C'})),
                                                    coords=dict(source_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['snk_tst_cell']:
                # i.e. the mean TST of upstream cells for a sink cell
                output['snk_tst_cell'] = xr.Dataset(data_vars=dict(upstream_tst=(['sink_cell', 'time'], ns_mx['snk_tst_cell'],
                                                                                 {'full_name': 'mean upstream TST',
                                                                                  'units': 'deg C',}),
                                                                   tst=(['sink_cell'], kwargs['tst'],
                                                                        {'full_name': 'sink cell TST',
                                                                         'units': 'deg C'})),
                                                    coords=dict(sink_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['src_cru_cell']:
                # i.e. the mean proportion of the time that a source cell IS NOT at bleaching risk, that sink cells ARE at bleaching risk
                output['src_cru_cell'] = xr.Dataset(data_vars=dict(downstream_cru=(['source_cell', 'time'], ns_mx['src_cru_cell'],
                                                                                   {'full_name': 'mean proportion of time source cell IS NOT at bleaching risk that sink cells ARE at bleaching risk',
                                                                                    'units': 'unitless',})),
                                                    coords=dict(source_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['snk_cru_cell']:
                # i.e. the mean proportion of the time that a sink cell IS at bleaching risk, that source cells ARE NOT at bleaching risk
                output['snk_cru_cell'] = xr.Dataset(data_vars=dict(upstream_cru=(['sink_cell', 'time'], ns_mx['snk_cru_cell'],
                                                                                 {'full_name': 'mean proportion of time sink cell IS at bleaching risk that source cells ARE NOT at bleaching risk',
                                                                                  'units': 'unitless',})),
                                                    coords=dict(sink_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1, hour=0),
                                                                                   periods=n_days, freq='D')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

            if export_mode['flux_src_cell']:
                output['flux_src_cell'] = xr.Dataset(data_vars=dict(ns=(['source_cell', 'sink_cell', 'time'], ns_mx['flux_src_cell'],
                                                                        {'full_name': 'number of settling larvae as a function of source time',
                                                                         'units': 'settling_larvae',}),
                                                                   rc=(['source_cell'], translate(cell_list, self.dicts['rc']),
                                                                       {'full_name': 'reef_cover',
                                                                        'units': 'm2'})),
                                                    coords=dict(source_cell=cell_list, sink_cell=cell_list,
                                                                time=pd.date_range(start=datetime(year=y0, month=1, day=1), periods=12, freq='M')),
                                                    attrs=dict(a=self.cfg['a']/conv_f,
                                                               b=self.cfg['b']/conv_f,
                                                               tc=self.cfg['tc']*conv_f,
                                                               μs=self.cfg['μs']/conv_f,
                                                               σ=self.cfg['σ'],
                                                               λ=self.cfg['λ']/conv_f,
                                                               ν=self.cfg['ν'],
                                                               configuration=self.cfg['preset'],
                                                               parcels_version=attr_dict['parcels_version'],
                                                               timestep_seconds=attr_dict['timestep_seconds'],
                                                               min_competency_seconds=attr_dict['min_competency_seconds'],
                                                               max_lifespan_seconds=attr_dict['max_lifespan_seconds'],
                                                               larvae_per_cell=attr_dict['larvae_per_cell'],
                                                               interp_method=attr_dict['interp_method'],
                                                               e_num=attr_dict['e_num'],
                                                               total_larvae_released=attr_dict['total_larvae_released'],
                                                               RK4_iterations=RK4_its))

        # Output test metrics
        if full_test:
            # Test individual fluxes
            ns_mx_sanitised = ns_array_testing.flatten()
            ns_mx_sanitised = ns_mx_sanitised[ns_mx_sanitised != 0]

            ns_mx_full_int_sanitised = ns_array_full_int_testing.flatten()
            ns_mx_full_int_sanitised = ns_mx_full_int_sanitised[ns_mx_full_int_sanitised != 0]

            ratio = ns_mx_sanitised/ns_mx_full_int_sanitised
            ratio = (ratio - 1)*100

            f, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.hist(ratio, bins=np.linspace(-100, 100, num=200), color='k')
            ax.set_title('Settling events')
            ax.set_xlabel('Semi-analytical v. full integration (difference, %)')
            ax.set_ylabel('Count')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.set_yscale('log')

            plt.savefig('settling_events_test_metric.pdf', bbox_inches='tight')

            if export_mode['src_str_cell']:
                # Test source strength
                ns_mx_sanitised = ns_mx['src_str_cell'].flatten()
                ns_mx_sanitised = ns_mx_sanitised[ns_mx_sanitised != 0]

                ns_mx_full_int_sanitised = ns_mx_full_int['src_str_cell'].flatten()
                ns_mx_full_int_sanitised = ns_mx_full_int_sanitised[ns_mx_full_int_sanitised != 0]

                ratio = ns_mx_sanitised/ns_mx_full_int_sanitised
                ratio = (ratio - 1)*100

                f, ax = plt.subplots(1, 1, figsize=(5, 5))

                ax.hist(ratio, bins=np.linspace(-100, 100, num=200), color='k')
                ax.set_title('Source strength')
                ax.set_xlabel('Semi-analytical v. full integration (difference, %)')
                ax.set_ylabel('Count')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.set_yscale('log')

                plt.savefig('src_str_cell_test_metric.pdf', bbox_inches='tight')

            if export_mode['flux_src_cell']:
                # Test flux matrix
                ns_mx_sanitised = ns_mx['flux_src_cell'].flatten()
                ns_mx_sanitised = ns_mx_sanitised[ns_mx_sanitised != 0]

                ns_mx_full_int_sanitised = ns_mx_full_int['flux_src_cell'].flatten()
                ns_mx_full_int_sanitised = ns_mx_full_int_sanitised[ns_mx_full_int_sanitised != 0]

                ratio = ns_mx_sanitised/ns_mx_full_int_sanitised
                ratio = (ratio - 1)*100

                f, ax = plt.subplots(1, 1, figsize=(5, 5))

                ax.hist(ratio, bins=np.linspace(-100, 100, num=200), color='k')
                ax.set_title('Flux')
                ax.set_xlabel('Semi-analytical v. full integration (difference, %)')
                ax.set_ylabel('Count')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.set_yscale('log')

                plt.savefig('flux_src_cell_test_metric.pdf', bbox_inches='tight')

        self.status['matrix'] = True

        print('...done!')
        print('##########################################################')
        print('')

        return output


class Matrix():
    """
    Initialise a larval dispersal matrix from Experiment output.
    To initialise, the first two arguments should be (1) the matrix and (2) a
    name for the matrix object.
    -----------
    Functions:
        TEST
    """

    def __init__(self, matrix, name):

        if isinstance(matrix, xr.Dataset):
            self.matrix = matrix
        else:
            raise TypeError('Matrix must be an xarray dataset.')

        if type(name) == str:
            self.name = name
        else:
            raise TypeError('Name must be a string.')

    def label(self, fh, **kwargs):
        '''
        Reorder and label a matrix based on an excel file

        Parameters
        ----------
        fh : String (file-handle for excel file)
        **kwargs : gen_labels : Bool (True to generate new labels file)
                   grid : String (file-handle for grid file. REQUIRED if gen_labels is True)

        '''

        # Assess whether matrix is cell or group based
        if 'source_cell' in self.matrix.coords or 'sink_cell' in self.matrix.coords:
            cell = True
            source_cell = True if 'source_cell' in self.matrix.coords else False
            sink_cell = True if 'sink_cell' in self.matrix.coords else False
        else:
            cell = False

        # If we are cell based, we need to firstly compute the group for each cell
        if cell:
            if 'gen_labels' in kwargs:
                if kwargs['gen_labels']:
                    grid = xr.open_dataset(kwargs['grid'])[['reef_idx_w', 'reef_cover_w']]
                    source_cell_grp = np.floor(self.matrix.source_cell/(2**8)).astype(int).values
                    _labels = pd.read_excel(fh)
                    cpg = []
                    for _gn in _labels['Group number']:
                        cpg.append(len(np.where(source_cell_grp == _gn)[0]))

                    self.labels = _labels.iloc[np.repeat(np.arange(len(_labels)), cpg)]
                    self.labels = self.labels.reset_index(drop=True)

                    # Now add cell number
                    self.labels['Cell number'] = np.zeros_like(self.labels['Group number'])

                    for row in self.labels.index.values:
                        if row == 0:
                            current_grp = self.labels.loc[row]['Group number']
                            grp_cnt = 0
                        else:
                            if self.labels.loc[row]['Group number'] == current_grp:
                                grp_cnt += 1
                            else:
                                current_grp = self.labels.loc[row]['Group number']
                                grp_cnt = 0

                        # Get cell number, and longitude and latitude
                        current_cell = self.matrix.source_cell[np.where(source_cell_grp == current_grp)].values.astype(int)[grp_cnt]
                        grid_idx = np.where(grid.reef_idx_w == current_cell)
                        self.labels.at[row, 'Cell number'] = current_cell
                        self.labels.at[row, 'Longitude'] = float(grid.lon_rho_w[grid_idx[1]])
                        self.labels.at[row, 'Latitude'] = float(grid.lat_rho_w[grid_idx[0]])
                        self.labels.at[row, 'Reef area'] = int(grid.reef_cover_w[grid_idx])

                    self.labels.to_csv('site_reference_cell.csv')
            else:
                self.labels = pd.read_csv(fh, index_col=0)

            if source_cell:
                self.matrix = self.matrix.reindex(source_cell=self.labels['Cell number'].values)
            if sink_cell:
                self.matrix = self.matrix.reindex(sink_cell=self.labels['Cell number'].values)
        else:
            self.labels = pd.read_excel(fh)
            self.matrix = self.matrix.reindex(source_group=self.labels['Group number'].values,
                                              sink_group=self.labels['Group number'].values)


        # Create country labels
        self.country_list = pd.unique(self.labels['Country']).astype(str)
        self.country_bnds = {}

        for country in self.country_list:
            self.country_bnds[country] = {'i0': self.labels.loc[self.labels['Country'] == country].index[0] + 0.5,
                                          'i1': self.labels.loc[self.labels['Country'] == country].index[-1] + 1.5,
                                          'mp': 0.5*(self.labels.loc[self.labels['Country'] == country].index[0] +
                                                     self.labels.loc[self.labels['Country'] == country].index[-1]) + 1,
                                          'nsites': len(self.labels.loc[self.labels['Country'] == country])}

        # Create L1 labels
        self.L1_list = pd.unique(self.labels['L1 Group']).astype(str)
        self.L1_bnds = {}

        for L1 in self.L1_list:
            self.L1_bnds[L1] = {'i0': self.labels.loc[self.labels['L1 Group'] == L1].index[0] + 0.5,
                                'i1': self.labels.loc[self.labels['L1 Group'] == L1].index[-1] + 1.5,
                                'mp': 0.5*(self.labels.loc[self.labels['L1 Group'] == L1].index[0] +
                                           self.labels.loc[self.labels['L1 Group'] == L1].index[-1]) + 1,
                                'nsites': len(self.labels.loc[self.labels['L1 Group'] == L1])}

        # Create dict to translate group number to L2/L1 group, country, and coordinates
        ref_label = 'Cell number' if cell else 'Group number'
        self.L2_dict = dict(zip(self.labels[ref_label], self.labels['L2 Group']))
        self.L1_dict = dict(zip(self.labels[ref_label], self.labels['L1 Group']))
        self.country_dict = dict(zip(self.labels[ref_label], self.labels['Country']))
        self.lon_dict = dict(zip(self.labels[ref_label], self.labels['Longitude']))
        self.lat_dict = dict(zip(self.labels[ref_label], self.labels['Latitude']))
        self.area_dict = dict(zip(self.labels[ref_label], self.labels['Reef area']))

        # Create group labels
        if cell:
            self.grp_bnds = {}
            for grp in np.unique(self.labels['Group number']):
                self.grp_bnds[grp] = {'i0': self.labels.loc[self.labels['Group number'] == grp].index[0] + 0.5,
                                      'i1': self.labels.loc[self.labels['Group number'] == grp].index[-1] + 0.5,
                                      'mp': 0.5*(self.labels.loc[self.labels['Group number'] == grp].index[0] +
                                                 self.labels.loc[self.labels['Group number'] == grp].index[-1]) + 1,
                                      'nsites': len(self.labels.loc[self.labels['Group number'] == grp]),
                                      'name': self.labels.loc[self.labels['Group number'] == grp]['L2 Group'].iloc[0]}

    def merge(self, merge_groups, **kwargs):
        '''
        Merge a list of nodes in the matrix

        Parameters
        ----------
        groups : list (of nodes)


        kwargs
        ------
        keep : Group to keep (must be an element of groups)
        L2 : New group L2 name
        L1 : New group L1 name
        country : New group country name

        '''

        # Check that all groups exist
        for group in merge_groups:
            if group not in self.matrix.sink_group:
                raise ValueError('Group ' + str(group) + ' not found.')

        # Get new list of group indices
        if 'keep' in kwargs:
            keep_group = int(kwargs['keep'])
            removal_list = [group for group in merge_groups if group != keep_group]
            assert len(removal_list) == len(merge_groups) - 1
        else:
            removal_list = merge_groups[1:]
            keep_group = merge_groups[0]

        # Create merged groups in temporary matrix
        temp_groups = self.matrix.source_group.values
        temp_groups[np.isin(temp_groups, merge_groups)] = keep_group
        self.matrix = self.matrix.assign_coords(source_group=temp_groups, sink_group=temp_groups)
        self.matrix = self.matrix.groupby('source_group').sum('source_group')
        self.matrix = self.matrix.groupby('sink_group').sum('sink_group')

        # Reorganise coordinates
        try:
            new_index = np.unique(self.labels['Group number'].values, return_index=True)[1]
            new_index = [self.labels['Group number'].values[index] for index in sorted(new_index)]
            self.matrix = self.matrix.reindex(source_group=new_index,
                                              sink_group=new_index)
        except:
            pass

        # Correct dictionaries
        if 'L2' in kwargs:
            self.L2_dict[keep_group] = kwargs['L2']

        if 'L1' in kwargs:
            self.L1_dict[keep_group] = kwargs['L1']

        if 'country' in kwargs:
            self.country_dict[keep_group] = kwargs['country']

        av_lon = 0
        av_lat = 0

        for group in removal_list:
            del self.L2_dict[group]
            del self.L1_dict[group]
            del self.country_dict[group]

            av_lon += (1/len(removal_list))*self.lon_dict[group]
            av_lat += (1/len(removal_list))*self.lat_dict[group]

            del self.lon_dict[group]
            del self.lat_dict[group]

        self.lon_dict[keep_group] = av_lon
        self.lat_dict[keep_group] = av_lat

def stream(flux, matrix_obj, **kwargs):
    '''
    This function outputs the sources or sinks for a site of interest in the
    provided time slice

    Parameters
    ----------
    flux : Flux matrix. Must have 'source_group' and 'sink_group' coordinates.
    matrix_obj: Matrix object associated with Flux matrix.
    **kwargs : source: int, source group
               sink: int, sink group
               num: int, number of (ranked) source/sink groups to export

    Returns
    -------
    Pandas dataframe

    '''

    if 'source' in kwargs:
        mode = 'source'
        group_num = kwargs['source']
    elif 'sink' in kwargs:
        mode = 'sink'
    else:
        raise Exception('Must either supply a source or sink.')

    if 'source' in kwargs and 'sink' in kwargs:
        raise Exception('Cannot supply both a source and a sink.')

    if 'num' in kwargs:
        num = kwargs['num']
    else:
        num = 0 # i.e. all nonzero

    if mode == 'source':
        flux2 = flux.loc[group_num, :].rename({'sink_group': 'group'})
    else:
        flux2 = flux.loc[:, group_num].rename({'source_group': 'group'})

    result = pd.DataFrame(index=flux2.group,
                          data=dict(flux=flux2,
                                    flux_frac=flux2/flux2.sum(),
                                    L2=[matrix_obj.L2_dict[key] for key in matrix_obj.L2_dict.keys()],
                                    L1=[matrix_obj.L1_dict[key] for key in matrix_obj.L1_dict.keys()],
                                    country=[matrix_obj.country_dict[key] for key in matrix_obj.country_dict.keys()],
                                    lon=[matrix_obj.lon_dict[key] for key in matrix_obj.lon_dict.keys()],
                                    lat=[matrix_obj.lat_dict[key] for key in matrix_obj.lat_dict.keys()],
                                    area=[matrix_obj.area_dict[key] for key in matrix_obj.area_dict.keys()]))

    result = result.sort_values(by=['flux'], ascending=False)

    if num > 0:
        result = result[:num]

    return result



















