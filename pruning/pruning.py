import math
import multiprocessing
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
import collections
from functools import reduce , partial
import operator
from multiprocessing.pool import Pool, ThreadPool
from .sorting import mask_network
from math import ceil
import gc
from concurrent.futures import ThreadPoolExecutor
from time import time
import numpy.ma as nma
from sys import getsizeof

def get_pruning_type(method):
    if method[:5] == 'unstr':
        pruning_type = 'unstr'
        method_identifier = method[6:]
    elif method[:3] == 'str':
        pruning_type = 'str'
        method_identifier = method[4:]
    else:
        pruning_type = 'str'
        method_identifier = method[4:]
    return pruning_type, method_identifier

def get_mask(array, prune_rate=0.1):
    array = np.abs(array)
    mask = np.ones(array.shape,  dtype=bool)
    k = int(np.prod(mask.shape) *prune_rate)
    # top k
    # idx = np.argpartition(array.reshape(-1), -k)[-k:]
    # smallest k
    idx = np.argpartition(array.reshape(-1), k)[:k]
    mask.reshape(-1)[idx] = 0
    return mask


def prune_act_unstruct(array, prune_rate=0.1, cursor=0.05):
    array_abs = -jnp.abs(array)
    k = int(array_abs.size * prune_rate)
    values, idxs = jax.lax.top_k(array_abs.reshape(-1), k)
    # return array.reshape(-1).at[idxs].set(0)
    return jnp.where(array_abs<cursor, 0, array)



def gen_mask_from_idxs(shape, idxs):
    mask = jnp.zeros(shape, dtype=bool)
    mask = mask.at[idxs].set(True)
    return mask


def top_k_1D(array, cand=1):
    values, idxs = jax.lax.top_k(jnp.abs(array), cand)
    mask = gen_mask_from_idxs(array.shape, idxs)
    # return array.at[idxs].set(0)
    return array * mask



def mute_values(array, cand, window, ref=None):
    if ref is None:
        ref = array
    values, idxs = jax.lax.top_k(jnp.abs(array) , cand)
    array = jnp.where(array>=jnp.min(values, axis=-1, keepdims=True) , ref, 0)
    return array



def prune_act_struct(array, prune_pattern=(1, 4), ref=None, prune_axis=-1):
    cand, window = prune_pattern
    ndim = len(array.shape)
    last_axis = ndim - 1
    to_swap_axes = False
    if prune_axis != -1:
        to_swap_axes = True
    if prune_axis < 0:
        prune_axis = ndim + prune_axis
    array = jnp.swapaxes(array, prune_axis, last_axis)
    ori_shape = array.shape
    ori_size = array.size
    new_shape = list(ori_shape[:-1]) + [ceil(ori_shape[-1]/window)] + [window]
    if ori_shape[-1]%window !=0:
        logging.warning(f'Ineffcient prune pattern. The window size [{window}] is not divisible to '
                        f'the last dimension [{ori_shape[-1]}]')
        new_size = reduce(operator.mul, new_shape, 1)
        array = jnp.pad(array.reshape(-1), (0, new_size - ori_size))
        array = array.reshape(new_shape)
        array = mute_values(array, cand, window, ref)
        array = array.reshape(-1)[:ori_size]
        array = array.reshape(ori_shape)
        if to_swap_axes:
            array = jnp.swapaxes(array, prune_axis, last_axis)
        return array
    else:
        array = array.reshape(new_shape)
        array = mute_values(array, cand, window, ref)
        array = array.reshape(ori_shape)
        if to_swap_axes:
            array = jnp.swapaxes(array, prune_axis, last_axis)
        return array







def get_mask_struct(array, prune_pattern=(1, 4), prune_axis=-1):
    mask = prune_act_struct(array, prune_pattern=prune_pattern, ref=1, prune_axis=prune_axis)
    return mask.astype(np.bool)




def prune_act_struct_deprecated_v2(array, prune_pattern=(1, 4)):
    cand, window = prune_pattern
    array_abs = jnp.abs(array)
    ori_shape = array.shape
    new_shape = list(ori_shape[:-1]) + [ceil(ori_shape[-1]//window)] + [window]
    array_abs = array_abs.reshape(new_shape)
    mask = mask_network(array_abs, window=window, cand=cand, take_abs=True, in_place=False)
    return array.at[mask.reshape(ori_shape)].set(0)




def flatten＿dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping) or isinstance(v, flax.core.FrozenDict):
            items.extend(flatten＿dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten＿dict_prettyprint(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping) or isinstance(v, flax.core.FrozenDict):
            items.extend(flatten＿dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v.shape))
    return dict(items)

def params_to_vector(params_list):
    new_params_list = []
    for params in params_list:
        new_params = params.reshape(-1)
        new_params_list.append(new_params)
    return np.concatenate(new_params_list)

def vector_to_params(params_vector, params_shape):
    params_list = []
    pointer = 0
    for shape in params_shape:
        params_list.append(params_vector[pointer:pointer+np.prod(shape)].reshape(shape))
        pointer += np.prod(shape)
    return params_list

def call_dict(keys, cur_dict):
    key = keys.popleft()
    next_dict = cur_dict[key]
    if isinstance(next_dict, collections.MutableMapping) or isinstance(next_dict, flax.core.FrozenDict):
        return call_dict(keys, next_dict)
    return cur_dict, key



def unified_local_pruning(state, prune_rate=0.1, sep='-',is_struct=False, prune_axis=-1):
    try:
        params = state.params
    except:
        params = state
    flatten_params = flatten＿dict(params, sep=sep)
    all_params = []
    all_params_shape = []
    all_params_mask_keys = []
    for k, v in flatten_params.items():
        if k[-6:] == 'kernel':
            all_params.append(v)
            all_params_shape.append(v.shape)
            all_params_mask_keys.append(k.split(sep))
    if is_struct:
        all_params_mask = [get_mask_struct(p, prune_pattern=prune_rate, prune_axis=prune_axis) for p in all_params]
    else:
        all_params_mask = [get_mask(p, prune_rate=prune_rate) for p in all_params]
    mask_key = [all_params_mask, all_params_mask_keys]
    return mask_key


def layered_local_pruning(state, layer_strategy=0.1, sep='-', is_struct=False, prune_axis=-1):
    try:
        params = state.params
    except:
        params = state
    all_params = []
    all_params_shape = []
    all_params_mask_keys = []
    prune_rates = []
    for keys, prune_rate in layer_strategy.items():
        v = getFromDict(params, keys.split(sep))
        all_params.append(v)
        all_params_shape.append(v.shape)
        all_params_mask_keys.append(keys.split(sep))
        prune_rates.append(prune_rate)
    if is_struct:
        all_params_mask = [get_mask_struct(p, prune_pattern=r, prune_axis=prune_axis) for p, r in zip(all_params, prune_rates)]
    else:
        all_params_mask = [get_mask(p, prune_rate=r) for p, r in zip(all_params, prune_rates)]
    mask_key = [all_params_mask, all_params_mask_keys]
    return mask_key





def global_pruning(state, prune_rate=0.1, sep='-',is_struct=False, prune_axis=-1):
    try:
        params = state.params
    except:
        params = state
    flatten_params = flatten＿dict(params, sep=sep)
    all_params = []
    all_params_shape = []
    all_params_mask_keys = []
    for k, v in flatten_params.items():
        if k[-6:] == 'kernel':
            all_params.append(v)
            all_params_shape.append(v.shape)
            all_params_mask_keys.append(k.split(sep))
    all_params = params_to_vector(all_params)
    if is_struct:
        all_params_mask = get_mask_struct(all_params, prune_pattern=prune_rate, prune_axis=prune_axis)
    else:
        all_params_mask = get_mask(all_params, prune_rate=prune_rate)
    all_params_mask = vector_to_params(all_params_mask, all_params_shape)
    mask_key = [all_params_mask, all_params_mask_keys]
    return mask_key


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def delInDict(dataDict, mapList):
    del getFromDict(dataDict, mapList[:-1])[mapList[-1]]


def update_strpruning_strategy(prune_strategy, epoch, eval_acc=0, eval_acc_bar=0, cur_density=0.5):
    if eval_acc > eval_acc_bar:
        latest_epoch = -1
        for prev_epoch in prune_strategy.keys():
            latest_epoch = max(latest_epoch, prev_epoch)
        cands, windows = prune_strategy[latest_epoch]['prune_rate_pattern']
        if cands != 1:
            new_cands = cands/2
            prune_strategy[epoch+1]['prune_rate_pattern'] = (new_cands, windows)
            print(f'[Pruning Strategy Update] from {(cands, windows)} to {(new_cands, windows)}')
        cur_density = cands/windows
    return cur_density


def pruning(state, epoch=None, mask_key=None, prune_strategy=None, method='local', type='unstr',  prune_rate_pattern=None, prune_axis=-1, float_pruning_start=-1, float_pruning_end=-1):
    is_pruned = False

    if prune_strategy is not None:
        if epoch in prune_strategy or -1 in prune_strategy or float_pruning_start<=epoch<=float_pruning_end:
            if -1 in prune_strategy:
                epoch = -1
            elif float_pruning_start<=epoch<=float_pruning_end:
                epoch = float_pruning_start
            method = prune_strategy[epoch]['method']
            prune_rate_pattern = prune_strategy[epoch]['prune_rate_pattern']
            is_pruned = True
    elif prune_rate_pattern is not None:
        method = method
        prune_rate_pattern = prune_rate_pattern
        is_pruned = True
    pruning_type, method_identifier = get_pruning_type(method)
    if pruning_type == 'unstr':
        is_struct = False
    else:
        is_struct = True
    if is_pruned:
        if method_identifier == 'global':
            mask_key = global_pruning(state, prune_rate=prune_rate_pattern, is_struct=is_struct, prune_axis=prune_axis)
        elif method_identifier == 'unified_local':
            mask_key = unified_local_pruning(state, prune_rate=prune_rate_pattern, is_struct=is_struct, prune_axis=prune_axis)
        elif method_identifier == 'local':
            mask_key = layered_local_pruning(state, layer_strategy=prune_rate_pattern, is_struct=is_struct, prune_axis=prune_axis)
    return is_pruned, mask_key




def set_mask_in_params_thread(mask, cur_elem):
    cur_elem *= mask
    return cur_elem

def get_params_section(params, all_params_mask_keys):
    return [ getFromDict(params, keys[:-1]) for keys in all_params_mask_keys],  [ getFromDict(params, keys) for keys in all_params_mask_keys]

def set_params_section(cur_elems, params_sections, key='kernel'):
    for p, c in zip(params_sections, cur_elems):
        if isinstance(p, flax.core.FrozenDict):
            p._dict[key]=c
        else:
            p[key]=c

pool = ThreadPool(multiprocessing.cpu_count())

def pruning_apply(state, mask_key, float_mask_val=0, input_is_state=True):
    if mask_key is None:
        return state, mask_key
    if input_is_state:
        params = flax.core.frozen_dict.unfreeze(state.params)
    else:
        params = flax.core.frozen_dict.unfreeze(state)
    all_params_mask, all_params_mask_keys = mask_key
    params_sections, cur_elems = get_params_section(params, all_params_mask_keys)
    # cur_elems = list(map(set_mask_in_params_thread, all_params_mask, cur_elems))
    cur_elems = pool.starmap(set_mask_in_params_thread, zip(all_params_mask, cur_elems))
    set_params_section(cur_elems, params_sections)
    if input_is_state:
        params = flax.core.frozen_dict.freeze(params)
        state = state.replace(params=params)
    else:
        params = flax.core.frozen_dict.freeze(params)
        state = params
    return state, mask_key


