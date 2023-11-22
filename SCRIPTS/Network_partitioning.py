#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script attempts to partition SECoW groups using Infomap clustering (using
the time-mean and temporal versions of the group flux matrix)
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import cmasher as cmr
import networkx as nx
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import sys
sys.path.insert(0, '../SIM/')
from SECoW import Matrix
from matplotlib.gridspec import GridSpec
from infomap import Infomap
from cartopy.feature import ShapelyFeature
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

###############################################################################
# PARAMETERS ##################################################################
###############################################################################

bio_code = sys.argv[1]
permitted_months = [1,2,3,10,11,12]
mean_length = 10   # Number of days to compute networks across
network_its = 1000 # Number of networks to initialise
plot_meow = True
n_cluster = 5

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../'
dirs['matrix'] = dirs['root'] + 'MATRICES/'
dirs['grid'] = dirs['root'] + 'GRID_DATA/'
dirs['fig'] = dirs['root'] + 'FIGURES/Partitioning/' + bio_code + '/'

# FILE-HANDLES
fh = {}
fh['matrix'] = dirs['matrix'] + 'WINDS_flux_src_grp_' + bio_code + '.nc'
fh['coral'] = dirs['grid'] + 'coral_grid.nc'
fh['grid'] = dirs['grid'] + 'griddata_winds.nc'
fh['site_list'] = dirs['grid'] + 'site_reference_grp_reordered.xlsx'
fh['meow'] = dirs['grid'] + 'MEOW/meow_ecos.shp'

if permitted_months == [1,2,3,10,11,12]:
    seas_str = 'full'
else:
    seas_str = 'M' + str(permitted_months).replace('[', '').replace(']', '').replace(' ', '').replace(',','-')

###############################################################################
# PREPROCESSING ###############################################################
###############################################################################

print('--------------------------------')
print('Preprocessing matrices...')
print('--------------------------------')

np.random.seed(1111)

# Preprocess connectivity matrix
with xr.open_dataset(fh['matrix'], chunks={'time': 2000}) as file:
    matrix = Matrix(file, bio_code)
    matrix.label(fh['site_list']) # For labelling (use later)
    group_list = matrix.matrix.source_group.values

    # Filter by months, and also extract the mean (filtered) flux matrix
    flux = matrix.matrix.ns[:, :, matrix.matrix.time.dt.month.isin(permitted_months)].compute()

    # Note - Infomap automatically carries out normalisation
    n_time = flux.shape[-1]
    n_groups = flux.shape[0]

# Store clustering for initialisations
infomap_clusters = np.zeros((len(group_list), network_its)) # (1 ... 180)

# Generate initialisations
np.random.seed(12345)
random_ts = np.random.choice(np.arange(n_time), size=(mean_length, network_its),
                             replace=True)

for it in tqdm(range(network_its), total=network_its):
    submatrix = flux[:, :, random_ts[:, it]].mean(dim='time')

    # Create graph
    graph = nx.from_numpy_array(submatrix.values,
                                create_using=nx.DiGraph)
    graph = nx.relabel_nodes(graph,
                             dict(zip(list(graph.nodes),
                                      list(submatrix.coords['source_group'].values))))

    # Import to Infomap
    infomap = Infomap(directed=True, teleportation_probability=0.15,
                      seed=1234, silent=True, num_trials=10, markov_time=1.0)
    mapping = infomap.add_networkx_graph(graph)
    mapping = {v: k for k, v in mapping.items()}
    infomap.run()

    # Save results
    results = infomap.get_modules(depth_level=6)
    infomap_clusters[:, it] = np.array([int(results[mapping[node_id]]) for node_id in np.arange(len(group_list))+1])

##############################################################################

# Carry out PCA on results
n_components = 3
pca = PCA(n_components=20, whiten=True)
pca_transform = pca.fit_transform(infomap_clusters)
pca_scaled = PCA(n_components=20, whiten=False)
pca_transform_scaled = pca_scaled.fit_transform(infomap_clusters)
pca_var = np.round(pca.explained_variance_ratio_*100, 1)

# Plot PCA variance
f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.stem(np.arange(20)+1, pca_var, linefmt='k-', markerfmt='ko', basefmt='none')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Proportion of variance (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0, np.max(pca_var)*1.1])
ax.set_xticks(np.arange(20) + 1)

plt.savefig(dirs['fig'] + bio_code + '_meta_cluster_pca_var_' + seas_str + '.pdf', bbox_inches='tight')
plt.close()

# Plot clustering in PCA space
x_norm = colors.Normalize(pca_transform[:, 0].min(),
                          pca_transform[:, 0].max())
y_norm = colors.Normalize(pca_transform[:, 1].min(),
                          pca_transform[:, 1].max())
z_norm = colors.Normalize(pca_transform[:, 2].min(),
                          pca_transform[:, 2].max())

f, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': '3d'})

# Project lines
for group in range(n_groups):
    ax.plot([pca_transform[group, 0], pca_transform[group, 0]],
            [pca_transform[group, 1], pca_transform[group, 1]],
            [-1.5, pca_transform[group, 2]], c='k', linewidth=0.2,
            alpha=0.2)

# num = 46 - 1 [where num is the group number of interest]
ax.scatter(pca_transform[:, 0],
           pca_transform[:, 1],
           pca_transform[:, 2],
           c=np.array([x_norm(pca_transform[:, 0]),
                       y_norm(pca_transform[:, 1]),
                       z_norm(pca_transform[:, 2]),
                       np.ones_like(pca_transform[:, 0])]).T,
           edgecolors='k', linewidths=0.5,
           marker='o', s=20)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([-1, 0, 1])

ax.set_xlabel('PC1 (' + str(pca_var[0]) + '%)', fontsize=10)
ax.set_ylabel('PC2 (' + str(pca_var[1]) + '%)', fontsize=10)
ax.set_zlabel('PC3 (' + str(pca_var[2]) + '%)', fontsize=10)

plt.savefig(dirs['fig'] + bio_code + '_meta_cluster_pca_' + seas_str + '.pdf', bbox_inches='tight')
plt.close()

# Carry out clustering
# Plot squared distances as a function of the number of clusters
inertia = []
for nc in range(1,13):
    inertia.append(KMeans(n_clusters=nc, n_init=100).fit(X=pca_transform_scaled[:,:3]).inertia_)

f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(range(1, 13), inertia, 'k-', marker='o')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')

plt.savefig(dirs['fig'] + bio_code + '_' + seas_str + '_kmeans_inertia.pdf', bbox_inches='tight')
plt.close()

# Actual plot
cluster_assignment = KMeans(n_clusters=n_cluster, n_init=100).fit_predict(X=pca_transform_scaled[:,:3])
f, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': '3d'})

# Project lines
for group in range(n_groups):
    ax.plot([pca_transform[group, 0], pca_transform[group, 0]],
            [pca_transform[group, 1], pca_transform[group, 1]],
            [-1.5, pca_transform[group, 2]], c='k', linewidth=0.2,
            alpha=0.2)

# num = 46 - 1 [where num is the group number of interest]
ax.scatter(pca_transform[:, 0],
           pca_transform[:, 1],
           pca_transform[:, 2],
           c=cluster_assignment,
           cmap=cmr.pride,
           vmin=0, vmax=n_cluster,
           edgecolors='k', linewidths=0.5,
           marker='o', s=20)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([-1, 0, 1])

ax.set_xlabel('PC1 (' + str(pca_var[0]) + '%)', fontsize=10)
ax.set_ylabel('PC2 (' + str(pca_var[1]) + '%)', fontsize=10)
ax.set_zlabel('PC3 (' + str(pca_var[2]) + '%)', fontsize=10)

plt.savefig(dirs['fig'] + bio_code + '_meta_cluster_pca_' + seas_str + '_kmeans' + str(n_cluster) + '.pdf', bbox_inches='tight')
plt.close()


# Compute PCA pairwise distance
pca_pos1, pca_pos2 = np.zeros((n_groups, n_groups, n_components)), np.zeros((n_groups, n_groups, n_components))
for component in range(n_components):
    pca_pos1[:, :, component], pca_pos2[:, :, component] = np.meshgrid(pca_transform_scaled[:, component],
                                                                       pca_transform_scaled[:, component])

pca_dist = xr.zeros_like(matrix.matrix.ns)[:, :, 0].drop('time')
pca_dist = pca_dist.assign_coords({'source_group': np.arange(n_groups)+1,
                                   'sink_group': np.arange(n_groups)+1})
pca_dist.data = np.linalg.norm(pca_pos1 - pca_pos2, axis=-1)
pca_dist = pca_dist.reindex(source_group=flux.source_group, sink_group=flux.sink_group)

axis_bnd = np.arange(n_groups+1) + 0.5

# Get a list of country border lines
country_border_pos = []
for country in matrix.country_list[1:]:
    country_border_pos.append(matrix.country_bnds[country]['i0'])

country_tick_pos = []
country_tick_label = []

for country in matrix.country_list:
    if matrix.country_bnds[country]['nsites'] > 1:
        country_tick_pos.append(matrix.country_bnds[country]['mp'])
        country_tick_label.append(country)

L1_border_pos = []
L1_border_pos.append([0])
for L1 in matrix.L1_list[1:]:
    L1_border_pos.append(matrix.L1_bnds[L1]['i0'])

country_border_pos.append(181)

L1_tick_pos = []
L1_tick_label_full = []
L1_tick_label = []

for L1 in matrix.L1_list:
    L1_tick_pos.append(matrix.L1_bnds[L1]['mp'])
    if matrix.L1_bnds[L1]['nsites'] > 1:
        L1_tick_label.append(L1)
    else:
        L1_tick_label.append('')

country_code = {'Maldives': 'Mal', 'Chagos': 'Cha', 'Mauritius': 'Mau', 'Seychelles': 'Sey',
                'Madagascar': 'Mad', 'France': 'Fra', 'Comoros': 'Com', 'Mozambique': 'Moz',
                'Tanzania': 'Tan', 'Kenya': 'Ken', 'Somalia': 'Som'}

L1_country_tick_label_ = [country_code[country_tick_label[np.argmax(country_border_pos > L1_border_pos[i])]] for i in range(len(L1_tick_label))]
L1_country_tick_label = []
ticker = 0

for i in range(len(L1_tick_label)):
    if i > 0:
        if L1_tick_label[i] == '':
            L1_country_tick_label.append('')
        else:
            if L1_country_tick_label_[i] == L1_country_tick_label_[i-1]:
                ticker += 1
            else:
                ticker = 0

            L1_country_tick_label.append(
                L1_country_tick_label_[i] + str(ticker + 1))
    else:
        L1_country_tick_label.append(
            L1_country_tick_label_[i] + str(ticker + 1))

f = plt.figure(constrained_layout=True, figsize=(14.5, 16.0))
gs = GridSpec(2, 1, figure=f, height_ratios=[1, 0.03], hspace=0.02)
ax = []
ax.append(f.add_subplot(gs[0, 0], zorder=2))  # Mean
ax.append(f.add_subplot(gs[0, 0], zorder=1))  # Mean (axis 2)
ax.append(f.add_subplot(gs[1, 0], zorder=1))  # Distance CB

# Plot mean data
gen_plot = ax[0].pcolormesh(axis_bnd, axis_bnd, pca_dist/pca_dist.max(),
                            vmin=0, vmax=1, cmap=cmr.pride_r)

for bndry in L1_border_pos:
    ax[0].plot([axis_bnd.min(), axis_bnd.max()], [
               bndry, bndry], 'w--', linewidth=0.5)
    ax[0].plot([bndry, bndry], [axis_bnd.min(),
               axis_bnd.max()], 'w--', linewidth=0.5)

for bndry in country_border_pos:
    ax[0].plot([axis_bnd.min(), axis_bnd.max()],
               [bndry, bndry], 'w-', linewidth=1)
    ax[0].plot([bndry, bndry], [axis_bnd.min(),
               axis_bnd.max()], 'w-', linewidth=1)

ax[0].set_aspect(1)
ax[0].set_facecolor('w')
ax[0].set_xticks(country_tick_pos)
ax[0].set_xticklabels(country_tick_label, rotation='vertical', fontsize=24,)
ax[0].set_yticks(country_tick_pos)
ax[0].set_yticklabels(country_tick_label, fontsize=24)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].xaxis.tick_top()
ax[0].set_xlim([0, 181])
ax[0].set_ylim([0, 181])
ax[0].invert_yaxis()
ax[0].set_xlabel('Destination', fontsize=36)
ax[0].set_ylabel('Source', fontsize=36)
ax[0].xaxis.set_label_position('top')

ax[1].set_aspect(1)
ax[1].set_xticks(L1_tick_pos)
ax[1].set_xticklabels(L1_country_tick_label, rotation='vertical', fontsize=16,)
ax[1].set_yticks(L1_tick_pos)
ax[1].set_yticklabels(L1_country_tick_label, fontsize=16)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].yaxis.tick_right()
ax[1].set_xlim([0, 181])
ax[1].set_ylim([0, 181])
ax[1].invert_yaxis()

cbar1 = plt.colorbar(gen_plot, cax=ax[2], orientation='horizontal')
cbar1.set_label('Normalised euclidean distance', size=22)
cbar1.ax.tick_params(axis='x', labelsize=18)

plt.savefig(dirs['fig'] + bio_code + '_meta_cluster_pca_dist_' + seas_str + '.pdf', bbox_inches='tight', dpi=300)
plt.close()

###############################################################################
# MA PLOTTING #################################################################
###############################################################################

f = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 1, figure=f, width_ratios=[1])
ax = []
ax.append(f.add_subplot(gs[0, 0],  projection = ccrs.PlateCarree())) # Network plot

# Set up grids
coral = xr.open_dataset(fh['coral'])
grid = xr.open_dataset(fh['grid'])

lon_bnd = coral.lon_psi_w.values
dlon = np.ediff1d(lon_bnd[:2])
lon_bnd = np.concatenate([lon_bnd[0]-dlon, lon_bnd, lon_bnd[-1]+dlon])

lat_bnd = coral.lat_psi_w.values
dlat = np.ediff1d(lat_bnd[:2])
lat_bnd = np.concatenate([lat_bnd[0]-dlat, lat_bnd, lat_bnd[-1]+dlat])

gl = []

# Draw background
ax[0].pcolormesh(lon_bnd, lat_bnd, coral.lsm_w.where(coral.lsm_w == 1),
                  vmin=0, vmax=1, cmap=cmr.neutral_r, rasterized=True, zorder=5)
bathymetry = ax[0].pcolormesh(lon_bnd, lat_bnd, grid.h, vmin=0, vmax=5250,
                              cmap=cmr.get_sub_cmap(cmr.neutral_r, 0, 0.2),
                              rasterized=True, zorder=1)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='k', linestyle='-',
                          alpha=0.5, zorder=15))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 5))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 5))
gl[0].top_labels = False
gl[0].right_labels = False
gl[0].ylabel_style = {'size': 8}
gl[0].xlabel_style = {'size': 8}

for node in np.arange(len(group_list))+1:
    ax[0].scatter(matrix.lon_dict[node],
                  matrix.lat_dict[node],
                  color=[x_norm(pca_transform[node-1, 0]),
                         y_norm(pca_transform[node-1, 1]),
                         z_norm(pca_transform[node-1, 2]),
                         1],
                  edgecolors='k', linewidth=0.2,
                  s=20, zorder=10)

# Add ecoregions
if plot_meow:
    reader = shpreader.Reader(fh['meow'])
    ecoregions = [ecr for ecr in reader.records()]

    for ecoregion in ecoregions:
        shape_feature = ShapelyFeature([ecoregion.geometry], ccrs.PlateCarree(),
                                        facecolor='None', edgecolor='k', lw=0.2, linestyle='-')
        ax[0].add_feature(shape_feature)

plt.savefig(dirs['fig'] + bio_code + '_meta_clusters_' + seas_str + '.pdf', bbox_inches='tight', dpi=1200)

# Kmeans version
f = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 1, figure=f, width_ratios=[1])
ax = []
ax.append(f.add_subplot(gs[0, 0],  projection = ccrs.PlateCarree())) # Network plot

# Set up grids
coral = xr.open_dataset(fh['coral'])
grid = xr.open_dataset(fh['grid'])

lon_bnd = coral.lon_psi_w.values
dlon = np.ediff1d(lon_bnd[:2])
lon_bnd = np.concatenate([lon_bnd[0]-dlon, lon_bnd, lon_bnd[-1]+dlon])

lat_bnd = coral.lat_psi_w.values
dlat = np.ediff1d(lat_bnd[:2])
lat_bnd = np.concatenate([lat_bnd[0]-dlat, lat_bnd, lat_bnd[-1]+dlat])

gl = []

# Draw background
ax[0].pcolormesh(lon_bnd, lat_bnd, coral.lsm_w.where(coral.lsm_w == 1),
                  vmin=0, vmax=1, cmap=cmr.neutral_r, rasterized=True, zorder=5)
bathymetry = ax[0].pcolormesh(lon_bnd, lat_bnd, grid.h, vmin=0, vmax=5250,
                              cmap=cmr.get_sub_cmap(cmr.neutral_r, 0, 0.2),
                              rasterized=True, zorder=1)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='k', linestyle='-',
                          alpha=0.5, zorder=15))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 5))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 5))
gl[0].top_labels = False
gl[0].right_labels = False
gl[0].ylabel_style = {'size': 8}
gl[0].xlabel_style = {'size': 8}

for node in np.arange(len(group_list))+1:
    ax[0].scatter(matrix.lon_dict[node],
                  matrix.lat_dict[node],
                  c=cluster_assignment[node-1],
                  cmap=cmr.pride,
                  vmin=0, vmax=n_cluster,
                  edgecolors='k', linewidth=0.2,
                  s=20, zorder=10)

ax[0].text(77, -23.15, '$k=$' + str(n_cluster), fontsize=16, va='bottom', ha='right')

plt.savefig(dirs['fig'] + bio_code + '_meta_clusters_' + seas_str + '_kmeans' + str(n_cluster) + '.pdf', bbox_inches='tight', dpi=1200)
plt.close()

# Plot the three principal components individually
f = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 1, figure=f, width_ratios=[1])
ax = []
ax.append(f.add_subplot(gs[0, 0],  projection = ccrs.PlateCarree())) # Network plot

# Set up grids
coral = xr.open_dataset(fh['coral'])
grid = xr.open_dataset(fh['grid'])

lon_bnd = coral.lon_psi_w.values
dlon = np.ediff1d(lon_bnd[:2])
lon_bnd = np.concatenate([lon_bnd[0]-dlon, lon_bnd, lon_bnd[-1]+dlon])

lat_bnd = coral.lat_psi_w.values
dlat = np.ediff1d(lat_bnd[:2])
lat_bnd = np.concatenate([lat_bnd[0]-dlat, lat_bnd, lat_bnd[-1]+dlat])

gl = []

# Draw background
ax[0].pcolormesh(lon_bnd, lat_bnd, coral.lsm_w.where(coral.lsm_w == 1),
                  vmin=0, vmax=1, cmap=cmr.neutral_r, rasterized=True, zorder=5)
bathymetry = ax[0].pcolormesh(lon_bnd, lat_bnd, grid.h, vmin=0, vmax=5250,
                              cmap=cmr.get_sub_cmap(cmr.neutral_r, 0, 0.2),
                              rasterized=True, zorder=1)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='k', linestyle='-',
                          alpha=0.5, zorder=15))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 5))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 5))
gl[0].top_labels = False
gl[0].right_labels = False
gl[0].ylabel_style = {'size': 8}
gl[0].xlabel_style = {'size': 8}

for node in np.arange(len(group_list))+1:
    ax[0].scatter(matrix.lon_dict[node],
                  matrix.lat_dict[node],
                  color=[x_norm(pca_transform[node-1, 0]),
                         0,
                         0,
                         1],
                  edgecolors='k', linewidth=0.2,
                  s=20, zorder=10)

# Add ecoregions
if plot_meow:
    reader = shpreader.Reader(fh['meow'])
    ecoregions = [ecr for ecr in reader.records()]

    for ecoregion in ecoregions:
        shape_feature = ShapelyFeature([ecoregion.geometry], ccrs.PlateCarree(),
                                        facecolor='None', edgecolor='k', lw=0.2, linestyle='-')
        ax[0].add_feature(shape_feature)

ax[0].text(77, -23.15, 'PC1 (' + str(pca_var[0]) + '%)', fontsize=16, va='bottom', ha='right')

plt.savefig(dirs['fig'] + bio_code + '_meta_clusters_' + seas_str + '_PC1.pdf', bbox_inches='tight', dpi=1200)
plt.close()

f = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 1, figure=f, width_ratios=[1])
ax = []
ax.append(f.add_subplot(gs[0, 0],  projection = ccrs.PlateCarree())) # Network plot

# Set up grids
coral = xr.open_dataset(fh['coral'])
grid = xr.open_dataset(fh['grid'])

lon_bnd = coral.lon_psi_w.values
dlon = np.ediff1d(lon_bnd[:2])
lon_bnd = np.concatenate([lon_bnd[0]-dlon, lon_bnd, lon_bnd[-1]+dlon])

lat_bnd = coral.lat_psi_w.values
dlat = np.ediff1d(lat_bnd[:2])
lat_bnd = np.concatenate([lat_bnd[0]-dlat, lat_bnd, lat_bnd[-1]+dlat])

gl = []

# Draw background
ax[0].pcolormesh(lon_bnd, lat_bnd, coral.lsm_w.where(coral.lsm_w == 1),
                  vmin=0, vmax=1, cmap=cmr.neutral_r, rasterized=True, zorder=5)
bathymetry = ax[0].pcolormesh(lon_bnd, lat_bnd, grid.h, vmin=0, vmax=5250,
                              cmap=cmr.get_sub_cmap(cmr.neutral_r, 0, 0.2),
                              rasterized=True, zorder=1)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='k', linestyle='-',
                          alpha=0.5, zorder=15))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 5))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 5))
gl[0].top_labels = False
gl[0].right_labels = False
gl[0].ylabel_style = {'size': 8}
gl[0].xlabel_style = {'size': 8}

for node in np.arange(len(group_list))+1:
    ax[0].scatter(matrix.lon_dict[node],
                  matrix.lat_dict[node],
                  color=[0,
                         y_norm(pca_transform[node-1, 1]),
                         0,
                         1],
                  edgecolors='k', linewidth=0.2,
                  s=20, zorder=10)

# Add ecoregions
if plot_meow:
    reader = shpreader.Reader(fh['meow'])
    ecoregions = [ecr for ecr in reader.records()]

    for ecoregion in ecoregions:
        shape_feature = ShapelyFeature([ecoregion.geometry], ccrs.PlateCarree(),
                                        facecolor='None', edgecolor='k', lw=0.2, linestyle='-')
        ax[0].add_feature(shape_feature)

ax[0].text(77, -23.15, 'PC2 (' + str(pca_var[1]) + '%)', fontsize=16, va='bottom', ha='right')
plt.savefig(dirs['fig'] + bio_code + '_meta_clusters_' + seas_str + '_PC2.pdf', bbox_inches='tight', dpi=1200)

f = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 1, figure=f, width_ratios=[1])
ax = []
ax.append(f.add_subplot(gs[0, 0],  projection = ccrs.PlateCarree())) # Network plot

# Set up grids
coral = xr.open_dataset(fh['coral'])
grid = xr.open_dataset(fh['grid'])

lon_bnd = coral.lon_psi_w.values
dlon = np.ediff1d(lon_bnd[:2])
lon_bnd = np.concatenate([lon_bnd[0]-dlon, lon_bnd, lon_bnd[-1]+dlon])

lat_bnd = coral.lat_psi_w.values
dlat = np.ediff1d(lat_bnd[:2])
lat_bnd = np.concatenate([lat_bnd[0]-dlat, lat_bnd, lat_bnd[-1]+dlat])

gl = []

# Draw background
ax[0].pcolormesh(lon_bnd, lat_bnd, coral.lsm_w.where(coral.lsm_w == 1),
                  vmin=0, vmax=1, cmap=cmr.neutral_r, rasterized=True, zorder=5)
bathymetry = ax[0].pcolormesh(lon_bnd, lat_bnd, grid.h, vmin=0, vmax=5250,
                              cmap=cmr.get_sub_cmap(cmr.neutral_r, 0, 0.2),
                              rasterized=True, zorder=1)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='k', linestyle='-',
                          alpha=0.5, zorder=15))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 5))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 5))
gl[0].top_labels = False
gl[0].right_labels = False
gl[0].ylabel_style = {'size': 8}
gl[0].xlabel_style = {'size': 8}

for node in np.arange(len(group_list))+1:
    ax[0].scatter(matrix.lon_dict[node],
                  matrix.lat_dict[node],
                  color=[0,
                         0,
                         z_norm(pca_transform[node-1, 2]),
                         1],
                  edgecolors='k', linewidth=0.2,
                  s=20, zorder=10)

# Add ecoregions
if plot_meow:
    reader = shpreader.Reader(fh['meow'])
    ecoregions = [ecr for ecr in reader.records()]

    for ecoregion in ecoregions:
        shape_feature = ShapelyFeature([ecoregion.geometry], ccrs.PlateCarree(),
                                        facecolor='None', edgecolor='k', lw=0.2, linestyle='-')
        ax[0].add_feature(shape_feature)
ax[0].text(77, -23.15, 'PC3 (' + str(pca_var[2]) + '%)', fontsize=16, va='bottom', ha='right')

plt.savefig(dirs['fig'] + bio_code + '_meta_clusters_' + seas_str + '_PC3.pdf', bbox_inches='tight', dpi=1200)