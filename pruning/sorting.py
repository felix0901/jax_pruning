import numpy as np
import jax.numpy as jnp
from functools import partial
import jax
sort_network = {
  4: [(1, 3), (0, 2), (0, 1), (2, 3), (1, 2)],
  6: [(0,5),(1,3),(2,4),(1,2),(3,4),(0,3),(2,5),(0,1),(2,3),(4,5),(1,2),(3,4)]
}

def compare_swap(a, b):
  if a[0] < b[0]:
    return a, b
  else:
    return b, a





def mask_network_1d(ori_array, window=4, cand=2, take_abs=False, in_place=True):
  series = sort_network[window]
  array = [(v, i) if not take_abs else (abs(v), i) for i, v in enumerate(ori_array) ]
  for a, b in series:
    array[a], array[b] = compare_swap(array[a], array[b])
  idxs = [array[i][1] for i in range(window)]
  if in_place:
    # ori_array[:] = ori_array[idxs]   # sorting
    ori_array[idxs[:cand]] = 0          # masking
    return ori_array
  else:
    mask = np.zeros((ori_array.shape), dtype=bool)
    mask[idxs[:cand]] = 1
    return mask

def mask_network(ori_array, window=4, cand=2, take_abs=False, in_place=True):
  mask_network_1d_func = partial(mask_network_1d, window=window, cand=cand, take_abs=take_abs, in_place=in_place)
  ret = np.apply_along_axis(mask_network_1d_func, axis=-1, arr=ori_array)
  return ret

# from jax._src.util import *
#
# def canonicalize_axis(axis, num_dims) -> int:
#   """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
#   axis = operator.index(axis)
#   if not -num_dims <= axis < num_dims:
#     raise ValueError(
#       "axis {} is out of bounds for array of dimension {}".format(
#         axis, num_dims))
#   if axis < 0:
#     axis = axis + num_dims
#   return axis
#
#
# def apply_along_axis(func1d, axis: int, arr, *args, **kwargs):
#   num_dims = len(arr.shape)
#   axis = canonicalize_axis(axis, num_dims)
#   func = lambda arr: func1d(arr, *args, **kwargs)
#   for i in range(1, num_dims - axis):
#     func = jax.vmap(func, in_axes=i, out_axes=-1)
#   for i in range(axis):
#     func = jax.vmap(func, in_axes=0, out_axes=0)
#   return func(arr)


if __name__ == '__main__':
  # array = [5, 7, 1, 3, 4, 2]
  array =  [[5, -7, -1, 4, 6, 2],
        [2, 1, 9, 8, 1, 3]]
  array = np.array(array)
  ret = mask_network(array, window=6, take_abs=True)
  print(ret)
