#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routine to compute CMIC
@author: Noam Vogt-Vincent (Mathematics from Ser-Giacomi et al.)
"""

import xarray as xr
import numpy as np
from tqdm import tqdm


def compute_CMEC(B_in, **kwargs):
    # Following Ser-Giacomi et al. (2021)
    # Note that this routine works for both forward and backward CMEC without
    # re-ordering if the time-slices are random. If the time-slices are NOT
    # random, then this routine by default only works for the forward CMEC. For
    # a backward CMEC, reverse the order of the time dimension first.

    # Also, remember that the input matrix needs to be transposed for the backward
    # case.

    # Get number of generations
    generations = B_in.shape[-1]

    # Output options
    full = kwargs['full'] if ('full' in kwargs) else False
    if full:
        gen_list = np.arange(1, generations+1)
    elif 'output_gen' in kwargs:
        gen_list = np.squeeze(np.array(kwargs['output_gen']))
    else:
        gen_list = np.array([generations])

    if 'source_group' in B_in.coords:
        src_node = 'source_group'
        snk_node = 'sink_group'
    else:
        src_node = 'source_cell'
        snk_node = 'sink_cell'

    assert np.max(gen_list) <= generations
    if generations > 200:
        print('Warning: this will take a while for large numbers of generations!')

    B = B_in.values
    L = (np.ones_like(B[:, :, 0]) - np.eye(B.shape[0])).astype(np.float32)
    output = np.zeros((len(gen_list), B.shape[0], B.shape[1]), dtype=np.float32)

    output[gen_list >= 1, :, :] += B[:, :, 0]

    if generations > 1:
        for generation in range(2, generations + 1):
            B_ = B[:, :, generation - 1]

            for it in range(1, generation):
                B_ = np.matmul(B[:, :, generation - 1 - it], L*B_)

            output[gen_list >= generation, :, :] += B_

    output = xr.DataArray(data=np.transpose(output, (1, 2, 0)),
                          dims=[src_node, snk_node, 'gen'],
                          coords={src_node: ([src_node], B_in[src_node].values),
                                  snk_node: ([snk_node], B_in[snk_node].values),
                                  'gen': (['gen'], gen_list)},
                          attrs=dict(full_name='Cumulative multistep explicit connectivity'))

    return output


def compute_CMIC(B_in, **kwargs):
    # Following Ser-Giacomi et al. (2021)
    # Note that this routine works for both forward and backward CMIC without
    # re-ordering if the time-slices are random. If the time-slices are NOT
    # random, then this routine by default only works for the forward CMIC. For
    # a backward CMIC, reverse the order of the time dimension first.

    # Also, remember that the input matrix needs to be transposed for the backward
    # case.

    # Get number of generations
    generations = B_in.shape[-1]

    # Output options
    full = kwargs['full'] if ('full' in kwargs) else False
    if full:
        gen_list = np.arange(1, generations+1)
    elif 'output_gen' in kwargs:
        gen_list = np.squeeze(np.array(kwargs['output_gen']))
    else:
        gen_list = np.array([generations])

    if 'source_group' in B_in.coords:
        src_node = 'source_group'
        snk_node = 'sink_group'
    else:
        src_node = 'source_cell'
        snk_node = 'sink_cell'

    assert np.max(gen_list) <= generations
    if generations > 200:
        print('Warning: this will take a while for large numbers of generations!')

    B = B_in.values
    L = (np.ones_like(B[:, :, 0]) - np.eye(B.shape[0])).astype(np.float32)
    output = np.zeros((len(gen_list), B.shape[0], B.shape[1]), dtype=np.float32)

    B_ = np.matmul(B[:, :, 0], B[:, :, 0].T)
    output[gen_list >= 1, :, :] += B_

    if generations > 1:
        for generation in range(2, generations + 1):
            B_ = np.matmul(B[:, :, generation - 1], B[:, :, generation - 1].T)

            for it in range(1, generation):
                B_ = np.matmul(np.matmul(B[:, :, generation - 1 - it], L*B_), B[:, :, generation - 1 - it].T)

            output[gen_list >= generation, :, :] += B_

    output = xr.DataArray(data=output.T,
                          dims=[src_node, snk_node, 'gen'],
                          coords={src_node: ([src_node], B_in[src_node].values),
                                  snk_node: ([snk_node], B_in[snk_node].values),
                                  'gen': (['gen'], gen_list)},
                          attrs=dict(full_name='Cumulative multistep implicit connectivity'))

    return output


def compute_CMIC_static(B_in, generations, **kwargs):
    # Following Ser-Giacomi et al. (2021)
    # Output options
    full = kwargs['full'] if ('full' in kwargs) else False
    if full:
        gen_list = np.arange(1, generations+1)
    elif 'output_gen' in kwargs:
        gen_list = np.squeeze(np.array(kwargs['output_gen']))
    else:
        gen_list = np.array([generations])

    if 'source_group' in B_in.coords:
        src_node = 'source_group'
        snk_node = 'sink_group'
    else:
        src_node = 'source_cell'
        snk_node = 'sink_cell'

    assert np.max(gen_list) <= generations
    if generations > 200:
        print('Warning: this will take a while for large numbers of generations!')

    B = B_in.values
    L = (np.ones_like(B) - np.eye(B.shape[0])).astype(np.float32)
    output = np.zeros((len(gen_list), B.shape[0], B.shape[1]), dtype=np.float32)

    pbar = tqdm(total=generations, unit='gen', )

    B_ = np.matmul(B, B.T)
    output[gen_list >= 1, :, :] += B_
    pbar.update(1)

    if generations > 1:
        for generation in range(2, generations+1):
            B_ = np.matmul(np.matmul(B, L*B_), B.T)
            output[gen_list >= generation, :, :] += B_
            pbar.update(1)

    output = xr.DataArray(data=output.T,
                          dims=[src_node, snk_node, 'gen'],
                          coords={src_node: ([src_node], B_in[src_node].values),
                                  snk_node: ([snk_node], B_in[snk_node].values),
                                  'gen': (['gen'], gen_list)},
                          attrs=dict(full_name='Cumulative multistep implicit connectivity'))

    return output


def compute_CMEC_static(B_in, generations, **kwargs):
    # Following Ser-Giacomi et al. (2021)
    # Output options
    full = kwargs['full'] if ('full' in kwargs) else False
    if full:
        gen_list = np.arange(1, generations+1)
    elif 'output_gen' in kwargs:
        gen_list = np.squeeze(np.array(kwargs['output_gen']))
    else:
        gen_list = np.array([generations])

    if 'source_group' in B_in.coords:
        src_node = 'source_group'
        snk_node = 'sink_group'
    else:
        src_node = 'source_cell'
        snk_node = 'sink_cell'

    assert np.max(gen_list) <= generations
    if generations > 200:
        print('Warning: this will take a while for large numbers of generations!')

    B = B_in.values
    L = (np.ones_like(B) - np.eye(B.shape[0])).astype(np.float32)
    output = np.zeros((len(gen_list), B.shape[0], B.shape[1]), dtype=np.float32)

    pbar = tqdm(total=generations, unit='gen', )

    B_ = B
    output[gen_list >= 1, :, :] += B_
    pbar.update(1)

    if generations > 1:
        for generation in range(2, generations+1):
            B_ = np.matmul(B, L*B_)
            output[gen_list >= generation, :, :] += B_
            pbar.update(1)

    output = xr.DataArray(data=np.transpose(output, (1, 2, 0)),
                          dims=[src_node, snk_node, 'gen'],
                          coords={src_node: ([src_node], B_in[src_node].values),
                                  snk_node: ([snk_node], B_in[snk_node].values),
                                  'gen': (['gen'], gen_list)},
                          attrs=dict(full_name='Cumulative multistep explicit connectivity'))

    return output

# Tests
if __name__ == "__main__":
    print('----------------------------')
    print('-      Running tests!      -')
    print('----------------------------')
    print('')

    # Testing against results in Ser-Giacomi et al. 2021

    network_a = np.array([[0., 0.6, 0.4, 0.],
                          [0., 0.5, 0., 0.5],
                          [0., 0.7, 0., 0.3],
                          [1., 0., 0., 0.,]])

    network_a_temporal = np.array([[[0., 0.6, 0.4, 0.],
                                    [0., 0.5, 0., 0.5],
                                    [0., 0.7, 0., 0.3],
                                    [1., 0., 0., 0]],
                                   [[0., 0.5, 0.5, 0.],
                                    [0., 0.5, 0., 0.5],
                                    [0., 0.6, 0., 0.4],
                                    [1., 0., 0., 0]],
                                   [[0., 0.4, 0.6, 0.],
                                    [0., 0.5, 0., 0.5],
                                    [0., 0.5, 0., 0.5],
                                    [1., 0., 0., 0]],
                                   [[0., 0.7, 0.3, 0.],
                                    [0., 0.5, 0., 0.5],
                                    [0., 0.8, 0., 0.2],
                                    [1., 0., 0., 0]],
                                   [[0., 0.8, 0.2, 0.],
                                    [0., 0.5, 0., 0.5],
                                    [0., 0.9, 0., 0.1],
                                    [1., 0., 0., 0]]]).transpose((1,2,0))

    network_b = np.array([[0., 0.7, 0.3, 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 1., 0.],
                          [1., 0., 0., 0.]])

    network_a_static = xr.DataArray(data=network_a,
                                    dims=['source_group', 'sink_group'],
                                    coords=dict(source_group=(['source_group'], np.arange(4)),
                                                sink_group=(['sink_group'], np.arange(4))))

    network_a_r1 = xr.DataArray(data=np.repeat(network_a[:, :, np.newaxis], 1, axis=-1),
                                dims=['source_group', 'sink_group', 'gen'],
                                coords=dict(source_group=(['source_group'], np.arange(4)),
                                            sink_group=(['sink_group'], np.arange(4)),
                                            gen=(['gen'], np.arange(1) + 1)))

    network_a_r5 = xr.DataArray(data=np.repeat(network_a[:, :, np.newaxis], 5, axis=-1),
                                dims=['source_group', 'sink_group', 'gen'],
                                coords=dict(source_group=(['source_group'], np.arange(4)),
                                            sink_group=(['sink_group'], np.arange(4)),
                                            gen=(['gen'], np.arange(5) + 1)))

    network_a_r100 = xr.DataArray(data=np.repeat(network_a[:, :, np.newaxis], 100, axis=-1),
                                  dims=['source_group', 'sink_group', 'gen'],
                                  coords=dict(source_group=(['source_group'], np.arange(4)),
                                              sink_group=(['sink_group'], np.arange(4)),
                                              gen=(['gen'], np.arange(100) + 1)))

    network_a_t5 = xr.DataArray(data=network_a_temporal,
                                dims=['source_group', 'sink_group', 'gen'],
                                coords=dict(source_group=(['source_group'], np.arange(4)),
                                            sink_group=(['sink_group'], np.arange(4)),
                                            gen=(['gen'], np.arange(5) + 1)))

    network_a_t100 = xr.DataArray(data=np.repeat(network_a_temporal, 20, axis=-1),
                                  dims=['source_group', 'sink_group', 'gen'],
                                  coords=dict(source_group=(['source_group'], np.arange(4)),
                                              sink_group=(['sink_group'], np.arange(4)),
                                              gen=(['gen'], np.arange(100) + 1)))

    network_b_static = xr.DataArray(data=network_b,
                                    dims=['source_group', 'sink_group'],
                                    coords=dict(source_group=(['source_group'], np.arange(4)),
                                                sink_group=(['sink_group'], np.arange(4))))

    network_b_r1 = xr.DataArray(data=np.repeat(network_b[:, :, np.newaxis], 1, axis=-1),
                                dims=['source_group', 'sink_group', 'gen'],
                                coords=dict(source_group=(['source_group'], np.arange(4)),
                                            sink_group=(['sink_group'], np.arange(4)),
                                            gen=(['gen'], np.arange(1) + 1)))

    network_b_r5 = xr.DataArray(data=np.repeat(network_b[:, :, np.newaxis], 5, axis=-1),
                                dims=['source_group', 'sink_group', 'gen'],
                                coords=dict(source_group=(['source_group'], np.arange(4)),
                                            sink_group=(['sink_group'], np.arange(4)),
                                            gen=(['gen'], np.arange(5) + 1)))

    network_b_r100 = xr.DataArray(data=np.repeat(network_b[:, :, np.newaxis], 100, axis=-1),
                                  dims=['source_group', 'sink_group', 'gen'],
                                  coords=dict(source_group=(['source_group'], np.arange(4)),
                                              sink_group=(['sink_group'], np.arange(4)),
                                              gen=(['gen'], np.arange(100) + 1)))

    # Test static CMIC
    # (1) Network A
    ans = np.array([[0.52, 0.3, 0.42, 0.],
                    [0.3, 0.5, 0.5, 0.],
                    [0.42, 0.5, 0.58, 0],
                    [0., 0., 0., 1.]], dtype=np.float32)
    assert np.allclose(compute_CMIC_static(network_a_static, 1, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMIC(network_a_r1, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.array([[0.84976, 0.67607, 0.755562, 0.6402272],
                    [0.67607, 0.73775, 0.73775, 0.57058],
                    [0.755562, 0.73775, 0.77971, 0.58398],
                    [0.6402272, 0.57058, 0.58398, 1.]])
    assert np.allclose(compute_CMIC_static(network_a_static, 5, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMIC(network_a_r5, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.ones((4,4))
    assert np.allclose(compute_CMIC_static(network_a_static, 100, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMIC(network_a_r100, full=False).data.squeeze(), ans, rtol=1e-03)

    # (2) Network B
    ans = np.array([[0.58, 0., 0.3, 0.],
                    [0., 1., 0., 0.],
                    [0.3, 0., 1., 0.],
                    [0., 0., 0., 1.]], dtype=np.float32)
    assert np.allclose(compute_CMIC_static(network_b_static, 1, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMIC(network_b_r1, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.array([[0.706, 0.153, 0.51, 0.2601],
                    [0.153, 1., 0.3, 0.153],
                    [0.51, 0.3, 1.0, 0.51],
                    [0.2601, 0.153, 0.51, 1.]])
    assert np.allclose(compute_CMIC_static(network_b_static, 5, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMIC(network_b_r5, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.ones((4,4))
    assert np.allclose(compute_CMIC_static(network_b_static, 100, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMIC(network_b_r100, full=False).data.squeeze(), ans, rtol=1e-03)

    # Test static CMEC
    # (1) Network A
    ans = np.array([[0., 0.6, 0.4, 0.],
                    [0., 0.5, 0., 0.5],
                    [0., 0.7, 0., 0.3],
                    [1., 0., 0., 0.,]], dtype=np.float32)
    assert np.allclose(compute_CMEC_static(network_a_static, 1, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMEC(network_a_r1, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.array([[0.855, 0.9856, 0.58, 0.9275],
                    [0.9375, 0.94, 0.35, 0.96875],
                    [0.9125, 0.964, 0.33, 0.95625],
                    [1.0, 0.952, 0.52, 0.855]], dtype=np.float32)
    assert np.allclose(compute_CMEC_static(network_a_static, 5, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMEC(network_a_r5, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.ones((4,4))
    assert np.allclose(compute_CMEC_static(network_a_static, 100, full=False).data.squeeze(), ans, 1e-03)
    assert np.allclose(compute_CMEC(network_a_r100, full=False).data.squeeze(), ans, rtol=1e-03)

    # (1) Network B
    ans = np.array([[0., 0.7, 0.3, 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 1., 0.],
                    [1., 0., 0., 0.]], dtype=np.float32)
    assert np.allclose(compute_CMEC_static(network_b_static, 1, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMEC(network_b_r1, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.array([[0.7, 0.7, 0.51, 0.7],
                    [1., 0.7, 0.3, 1.0],
                    [0., 0., 1.0, 0.],
                    [1.0, 0.7, 0.51, 0.7]], dtype=np.float32)
    assert np.allclose(compute_CMEC_static(network_b_static, 5, full=False).data.squeeze(), ans, rtol=1e-03)
    assert np.allclose(compute_CMEC(network_b_r5, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.array([[0.7, 0.7, 1.0, 0.7],
                    [1., 0.7, 1.0, 1.0],
                    [0., 0., 1.0, 0.],
                    [1.0, 0.7, 1.0, 0.7]], dtype=np.float32)
    assert np.allclose(compute_CMEC_static(network_b_static, 100, full=False).data.squeeze(), ans, 1e-03)
    assert np.allclose(compute_CMEC(network_b_r100, full=False).data.squeeze(), ans, rtol=1e-03)

    # Test temporal CMIC
    ans = np.array([[0.52, 0.3, 0.42, 0.],
                    [0.3, 0.5, 0.5, 0.],
                    [0.42, 0.5, 0.58, 0],
                    [0., 0., 0., 1.]], dtype=np.float32)
    assert np.allclose(compute_CMIC(network_a_r1, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.array([[0.8705, 0.7102, 0.7829, 0.6438],
                    [0.7102, 0.766, 0.766, 0.5581],
                    [0.78296, 0.766, 0.80344, 0.5869],
                    [0.6438, 0.5581, 0.5869, 1.0]], dtype=np.float32)
    assert np.allclose(compute_CMIC(network_a_t5, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.ones((4,4))
    assert np.allclose(compute_CMIC(network_a_r100, full=False).data.squeeze(), ans, rtol=1e-03)

    # Test temporal CMEC
    ans = np.array([[0., 0.6, 0.4, 0.],
                    [0., 0.5, 0., 0.5],
                    [0., 0.7, 0., 0.3],
                    [1., 0., 0., 0.,]], dtype=np.float32)
    assert np.allclose(compute_CMEC(network_a_r1, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.array([[0.865, 0.9952, 0.52, 0.9325],
                    [0.9375, 0.94, 0.4, 0.96875],
                    [0.9125, 0.964, 0.32, 0.956],
                    [1.0, 0.95, 0.55, 0.875]], dtype=np.float32)
    assert np.allclose(compute_CMEC(network_a_t5, full=False).data.squeeze(), ans, rtol=1e-03)

    ans = np.ones((4,4))
    assert np.allclose(compute_CMEC(network_a_t100, full=False).data.squeeze(), ans, rtol=1e-03)

