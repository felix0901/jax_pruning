JAX-Pruning: A JAX implementation of structure and unstructure pruning
===
This is an implementation of Jax-based pruning module.

We support two kinds of pruning target: 
* weight pruning
* activation pruning

We support two kinds of pruning pattern:
* Unstructured pruning
* Structured pruning

---------------

# Weight Pruning
Weight pruning involves two functions
* ``pruning``: It generates the pruning mask.
* ``pruning_apply``: It applies the pruning mask onto the weight.

We show an example on how to use it by a snapshot of code below:
```python
# Training  
state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    config.batch_size,
                                                    input_rng)
# Generate pruning mask
mask_updated, mask_key = pruning(state, epoch, mask_key=mask_key, prune_strategy=pruning_strategy)
# Apply pruning mask
state, mask_key = pruning_apply(state, mask_key)
```


## Pruning strategy
We use ``pruning_strategy`` to define **pruning pattern**, **pruning layer**, and **pruning schedule**.
### pruning rate or pruning pattern (``prune_rate_pattern``)
* Unstructure pruning (``unstr``): We specify the target density value of the layer. E.g., density=0.9 will prune 10% of the weight of the layer.
  * Use the 
* Structure pruning (``str``): We specify a structure sparsity_pattern. The structure sparsity_pattern is defined as (N, M) or (N:M), which means out of M consecutive elements only N elements are non-zeros.
  * E.g., (2, 4) has density of 50%
  * E.g., (1, 4) has density of 25%

### pruning layer
We support three kinds of method to specify the pruning layer:
* Sparify all layers to the same ``prune_rate_pattern``: use the key word ``unified_local``.
* Sparify all layers while the entire model achieve the ``prune_rate_pattern``: use the key word ``global``.
  * This is different from previous one. Here, differnt layers can have differnt density as long as the entire achieve the density level specify by ``prune_rate_pattern``
* Detail layer-by-layer specifications of the specific ``prune_rate_pattern`` for each of the layer: use the key word ``local`` and use the ``layer-name`` to specify the layer.
### pruning schedule
* We allow users to define any pruning schedule by specifying which epoch to prune the model to how much sparsity (structure or unstructure).
 
  The generated sparse mask will be maintained after pruning.  

  * Example 1, assuming total training epochs=100 epochs. Gradual pruning at epoch 10, 20, 30, ..., 90. Between epoch 0-10 dense training. Between epoch 10-20 fine-tuning while keeping the mask calculated at epoch-10 to recover the accuracy loss.
    * (epoch-10, 0.9), (epoch-20, 0.8), (epoch-30, 0.7), .., (epoch-90, 0.1)
  
  * Example 2, assuming total training epochs=100 epochs. Dense training for 50 epochs, sparsify with structure pruning with sparsity pattern of (2, 4) at epoch-50, and fine-tuning for 50 epochs with the structured mask found at epoch-50 to recover the accuracy. 
    *  (epoch-50, (2, 4))
  
### We show examples of ``pruning strategy`` as follows:
Example-1 
* Pruning schedule: ``1`` --> Pruning at the first epoch
* Pruning layer: ``global`` --> Prune all the layers while overall the entire model achieve the density level specified by ``prune_rate_pattern``
* Pruning rate or pattern: ``unstr`` ``0.9`` --> It use unstructure pruning and the density is 90%.
```python
pruning_strategy = {1: {'method': 'unstr_global', 
                        'prune_rate_pattern':0.9
                        } 
                    }
```
Example-2
* Pruning schedule: ``10`` --> Pruning at the 10th epoch
* Pruning layer: ``unified_local`` --> Prune all the layers to the same density level specified by ``prune_rate_pattern``
* Pruning rate or pattern: ``str`` `` (2, 4)`` --> It use structure pruning and the pruning pattern is (2, 4).
```python
pruning_strategy = {10: {'method': 'str_unified_local', 
                        'prune_rate_pattern': (2, 4)
                        }
                    }
```
Example-3

At epoch-10:
  * Pruning schedule: ``10`` --> Pruning at the 10th epoch
  * Pruning layer: ``unified_local`` --> Prune all the layers to the same density level specified by ``prune_rate_pattern``
  * Pruning rate or pattern: ``str`` `` (2, 4)`` --> It use structure pruning and the pruning pattern is (2, 4).
  
At epoch-20:
  * Pruning schedule: ``20`` --> Pruning at the 20th epoch
  * Pruning layer: ``unified_local`` --> Prune all the layers to the same density level specified by ``prune_rate_pattern``
  * Pruning rate or pattern: ``str`` `` (1, 4)`` --> It use structure pruning and the pruning pattern is (1, 4).
```python
pruning_strategy = {10: {'method': 'str_unified_local', 
                        'prune_rate_pattern': (2, 4)
                        },
                    20: {'method': 'str_unified_local',
                         'prune_rate_pattern': (1, 4)
                         }
                    }
```

Example-4
* Pruning schedule: ``10`` --> Pruning at the 10th epoch
* Pruning layer: ``local`` --> We will only prune a subset of the layers
* Pruning rate or pattern: ``unstr``  --> We will use unstructure pruning
* More specific layer-by-layer strategy:
  * For layer ``Dense_0``, we use density ``0.9``
  * For layer ``Dense_1``, we use density ``0.8``
    * Here we add one ``kernel`` to specify we only want to sparsify the ``weight kernel`` while keeping the ``bias`` dense 
```python
pruning_strategy = {10: {'method': 'unstr_unified_local', 
                        'prune_rate_pattern': {'Dense_0-kernel': 0.9,
                                               'Dense_1-kernel': 0.8,
                                               }
                        }
                    }
```



------------------
# Activation Pruning
They are two types of pruning pattern unstructure and structure, which correspond to two different functions:
* ``puning_act_unstruct``: Meaning pruning function for activation pruning with unstructure pattern.
  It can be used as follows:
  ```python
  # 0 < density <= 1
  x = prune_act_unstruct(x, density)
  ```
  E.g.,
  ```python
  x = prune_act_unstruct(x, 0.9)
  ```
* ``pruning_act_struct``: Meaning pruning function for activation pruning with structure pattern.
  It can be used as follows:
  ```python
  x = pruning_act_struct(x, sparsity_pattern)
  ```
  E.g.,
  ```python
  x = pruning_act_struct(x, (2, 4))
  ```
  

These two functions can be inserted in the model as shown in the follows
```python
class CNN(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = prune_act_unstruct(x, 0.9)    # Pruning x with unstructure sparsity with density of 90%
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = prune_act_struct(x, (2, 4)) # Pruning x with structure sparsity with sparsity pattern of (2, 4), whose density is 50%
    return x
```

------------

# A complete example
We grab an mnist example from [google/flax](https://github.com/google/flax/tree/main/examples/mnist) and inserted the above mentioned four functions in the code to demonstrate how to use them.

A snap shot of code where we insert the pruning is as follows:

At the training loops
```python
 for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    config.batch_size,
                                                    input_rng)
    state, mask_key = pruning_apply(state, mask_key)
    mask_updated, mask_key = pruning(state, epoch, mask_key=mask_key, prune_strategy=pruning_strategy)
    if mask_updated:
      state, mask_key = pruning_apply(state, mask_key)
    _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                              test_ds['label'])
```

At model definition
```python
```python
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = prune_act_unstruct(x, 0.9)    # Pruning x with unstructure sparsity with density of 90%
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = prune_act_struct(x, (2, 4)) # Pruning x with structure sparsity with sparsity pattern of (2, 4), whose density is 50%
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x
```



To see the complete code, please see [mnist example](example/mnist/train.py).


-------------


## Citation
```
@software{jax_pruning,
  author = {Kao, Sheng-Chun},
  title = {{JAX-Pruning: A JAX implementation of structure and unstructure pruning}},
  url = {https://github.com/felix0901/jax_pruning},
  version = {0.1},
  year = {2022}
}
```
