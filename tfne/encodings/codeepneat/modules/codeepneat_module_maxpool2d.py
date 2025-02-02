from __future__ import annotations

import math
import random
import statistics

import numpy as np
import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase
from tfne.helper_functions import round_with_step


class CoDeepNEATModuleMaxPool2D(CoDeepNEATModuleBase):
    """
    TFNE CoDeepNEAT module encapsulating a MaxPool2D layer. The downsampling layer is another Conv2D layer.
    """

    def __init__(self,
                 config_params,
                 module_id,
                 parent_mutation,
                 dtype,
                 merge_method=None,
                 pool_size=None,
                 strides=None,
                 padding=None,
                 self_initialization_flag=False):
        """
        Create module by storing supplied parameters. If self initialization flag is supplied, randomly initialize the
        module parameters based on the range of parameters allowed by config_params
        @param config_params: dict of the module parameter range supplied via config
        @param module_id: int of unique module ID
        @param parent_mutation: dict summarizing the mutation of the parent module
        @param dtype: string of deserializable TF dtype
        @param merge_method: dict representing a TF deserializable merge layer
        @param pool_size: see TF documentation
        @param strides: see TF documentation
        @param padding: see TF documentation
        @param self_initialization_flag: bool flag indicating if all module parameters should be randomly initialized
        """
        # Register the implementation specifics by calling parent class
        super().__init__(config_params, module_id, parent_mutation, dtype)

        # Register the module parameters
        self.merge_method = merge_method
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        # If self initialization flag is provided, initialize the module parameters as they are currently set to None
        if self_initialization_flag:
            self._initialize()

    def __str__(self) -> str:
        """
        @return: string representation of the module
        """
        return "CoDeepNEAT MaxPool2D Module | ID: {:>6} | Fitness: {:>6} | Pool Size: {:>6} | Strides: {:>6} | Padding: {:>6}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    str(self.pool_size),
                    str(self.strides),
                    self.padding)

    def _initialize(self):
        """
        Randomly initialize all parameters of the module based on the range of parameters allowed by the config_params
        variable.
        """
        # Uniform randomly set module parameters
        self.merge_method = random.choice(self.config_params['merge_method'])
        self.merge_method['config']['dtype'] = self.dtype
        self.pool_size = (random.randint(int(self.config_params['pool_size']['min']), 
                                         int(self.config_params['pool_size']['max'])), 
                          random.randint(int(self.config_params['pool_size']['min']),
                                         int(self.config_params['pool_size']['max'])))
        self.strides = (random.randint(int(self.config_params['strides']['min']), 
                                         int(self.config_params['strides']['max'])), 
                          random.randint(int(self.config_params['strides']['min']),
                                         int(self.config_params['strides']['max'])))
        self.padding = random.choice(self.config_params['padding'])

    def create_module_layers(self) -> (tf.keras.layers.Layer, ...):
        """
        Instantiate TF layers with their respective configuration that are represented by the current module
        configuration. Return the instantiated module layers in their respective order as a tuple.
        @return: tuple of instantiated TF layers represented by the module configuration.
        """
        # Create iterable that contains all layers concatenated in this module
        module_layers = list()

        max_pool_layer = tf.keras.layers.MaxPool2D(pool_size=self.pool_size,
                                                    strides=self.strides,
                                                    padding=self.padding,
                                                    dtype=self.dtype)
        module_layers.append(max_pool_layer)

        # Return the iterable containing all layers present in the module
        return module_layers

    def create_downsampling_layer(self, in_shape, out_shape) -> tf.keras.layers.Layer:
        """
        Create Conv2D layer that downsamples the non compatible input shape to a compatible input shape of the module
        @param in_shape: int tuple of incompatible input shape
        @param out_shape: int tuple of the intended output shape of the downsampling layer
        @return: instantiated TF Conv2D layer that can downsample in_shape to out_shape
        """
        # As the MaxPool2D module downsamples with a Conv2D layer, assure that the input and output shape
        # are of dimension 4 and that the second and third channel are identical
        if not (len(in_shape) == 4 and len(out_shape) == 4):
            raise NotImplementedError(f"Downsampling Layer for the shapes {in_shape} and {out_shape}, not having 4 "
                                      f"channels has not yet been implemented for the Dropout module")

        if in_shape[1] != in_shape[2] or out_shape[1] != out_shape[2]:
            filters = in_shape[3]
            kernel_size = in_shape[1] - out_shape[1] + 1
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=None,
                                          dtype=self.dtype)

        # If Only the second and thid channel have to be downsampled then carry over the size of the fourth channel and
        # adjust the kernel size to result in the adjusted second and third channel size
        if out_shape[1] is not None and out_shape[3] is None:
            filters = in_shape[3]
            kernel_size = in_shape[1] - out_shape[1] + 1
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=None,
                                          dtype=self.dtype)

        # If Only the fourth channel has to be downsampled then carry over the size of the second and fourth channel and
        # adjust the filters to result in the adjusted fourth channel size
        elif out_shape[1] is None and out_shape[3] is not None:
            filters = out_shape[3]
            kernel_size = in_shape[1]
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(1, 1),
                                          padding='same',
                                          activation=None,
                                          dtype=self.dtype)

        # If the second, third and fourth channel have to be downsampled adjust both the filters and kernel size
        # accordingly to result in the desired output shape
        elif out_shape[1] is not None and out_shape[3] is not None:
            filters = out_shape[3]
            kernel_size = in_shape[1] - out_shape[1] + 1
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=None,
                                          dtype=self.dtype)
        else:
            raise RuntimeError(f"Downsampling to output shape {out_shape} from input shape {in_shape} not possible"
                               f"with a Conv2D layer")

    def create_mutation(self,
                        offspring_id,
                        max_degree_of_mutation) -> CoDeepNEATModuleMaxPool2D:
        """
        Create mutated MaxPool2D module and return it. Categorical parameters are chosen randomly from all
        available values. Sortable parameters are perturbed through a random normal distribution with the current value
        as mean and the config specified stddev
        @param offspring_id: int of unique module ID of the offspring
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated MaxPool2D module with mutated parameters
        """
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'pool_size': self.pool_size,
                            'strides': self.strides,
                            'padding': self.padding}

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 4)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(4), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                offspring_params['merge_method'] = random.choice(self.config_params['merge_method'])
                parent_mutation['mutated_params']['merge_method'] = self.merge_method
            elif param_to_mutate == 1:
                perturbed_pool_size_x = round_with_step(np.random.normal(loc=self.pool_size[0],
                                                            scale=self.config_params['pool_size']['stddev']),
                                                        self.config_params['pool_size']['min'],
                                                        self.config_params['pool_size']['max'],
                                                        self.config_params['pool_size']['step'])
                perturbed_pool_size_y = round_with_step(np.random.normal(loc=self.pool_size[1],
                                                            scale=self.config_params['pool_size']['stddev']),
                                                        self.config_params['pool_size']['min'],
                                                        self.config_params['pool_size']['max'],
                                                        self.config_params['pool_size']['step'])
                offspring_params['pool_size'] = (int(perturbed_pool_size_x), int(perturbed_pool_size_y))
                parent_mutation['mutated_params']['pool_size'] = self.pool_size
            elif param_to_mutate == 2:
                perturbed_strides_x = round_with_step(np.random.normal(loc=self.strides[0],
                                                            scale=self.config_params['strides']['stddev']),
                                                        self.config_params['strides']['min'],
                                                        self.config_params['strides']['max'],
                                                        self.config_params['strides']['step'])
                perturbed_strides_y = round_with_step(np.random.normal(loc=self.strides[1],
                                                            scale=self.config_params['strides']['stddev']),
                                                        self.config_params['strides']['min'],
                                                        self.config_params['strides']['max'],
                                                        self.config_params['strides']['step'])
                offspring_params['strides'] = (int(perturbed_strides_x), int(perturbed_strides_y))
                parent_mutation['mutated_params']['strides'] = self.strides
            elif param_to_mutate == 3:
                offspring_params['padding'] = random.choice(self.config_params['padding'])
                parent_mutation['mutated_params']['padding'] = self.padding

        return CoDeepNEATModuleMaxPool2D(config_params=self.config_params,
                                                      module_id=offspring_id,
                                                      parent_mutation=parent_mutation,
                                                      dtype=self.dtype,
                                                      **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> CoDeepNEATModuleMaxPool2D:
        """
        Create crossed over MaxPool2D module and return it. Carry over parameters of fitter parent for
        categorical parameters and calculate parameter average between both modules for sortable parameters
        @param offspring_id: int of unique module ID of the offspring
        @param less_fit_module: second MaxPool2D module with lower fitness
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated MaxPool2D module with crossed over parameters
        """
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        if random.random() <= 0.5:
            offspring_params['merge_method'] = self.merge_method
        else:
            offspring_params['merge_method'] = less_fit_module.merge_method
        
        offspring_params['pool_size'] = ((self.pool_size[0] + less_fit_module.pool_size[0]) // 2, (self.pool_size[1] + less_fit_module.pool_size[1]) // 2)
        offspring_params['strides'] = ((self.strides[0] + less_fit_module.strides[0]) // 2, (self.strides[1] + less_fit_module.strides[1]) // 2)
        
        if random.random() <= 0.5:
            offspring_params['padding'] = self.padding
        else:
            offspring_params['padding'] = less_fit_module.padding

        return CoDeepNEATModuleMaxPool2D(config_params=self.config_params,
                                                      module_id=offspring_id,
                                                      parent_mutation=parent_mutation,
                                                      dtype=self.dtype,
                                                      **offspring_params)

    def serialize(self) -> dict:
        """
        @return: serialized constructor variables of the module as json compatible dict
        """
        return {
            'module_type': self.get_module_type(),
            'module_id': self.module_id,
            'parent_mutation': self.parent_mutation,
            'merge_method': self.merge_method,
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        }

    def get_distance(self, other_module) -> float:
        """
        Calculate distance between 2 MaxPool2D modules by inspecting each parameter, calculating the
        congruence between each and eventually averaging the out the congruence. The distance is returned as the average
        congruences distance to 1.0. The congruence of continuous parameters is calculated by their relative distance.
        The congruence of categorical parameters is either 1.0 in case they are the same or it's 1 divided to the amount
        of possible values for that specific parameter. Return the calculated distance.
        @param other_module: second MaxPool2D module to which the distance has to be calculated
        @return: float between 0 and 1. High values indicating difference, low values indicating similarity
        """
        if not isinstance(other_module, CoDeepNEATModuleMaxPool2D):
            return 0.0
    
        congruence_list = list()
        if self.merge_method == other_module.merge_method:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['merge_method']))
        if self.pool_size == other_module.pool_size:
            congruence_list.append(1.0)
        else:
            if self.pool_size[0] >= other_module.pool_size[0]:
                congruence_list.append(other_module.pool_size[0] / self.pool_size[0])
            else:
                congruence_list.append(self.pool_size[0] / other_module.pool_size[0])
            if self.pool_size[1] >= other_module.pool_size[1]:
                congruence_list.append(other_module.pool_size[1] / self.pool_size[1])
            else:
                congruence_list.append(self.pool_size[1] / other_module.pool_size[1])
        if self.strides == other_module.strides:
            congruence_list.append(1.0)
        else:
            if self.pool_size[0] >= other_module.pool_size[0]:
                congruence_list.append(other_module.pool_size[0] / self.pool_size[0])
            else:
                congruence_list.append(self.pool_size[0] / other_module.pool_size[0])
            if self.pool_size[1] >= other_module.pool_size[1]:
                congruence_list.append(other_module.pool_size[1] / self.pool_size[1])
            else:
                congruence_list.append(self.pool_size[1] / other_module.pool_size[1])
        if self.padding == other_module.padding:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['padding']))

        # Return the distance as the distance of the average congruence to the perfect congruence of 1.0
        return round(1.0 - statistics.mean(congruence_list), 4)

    def get_module_type(self) -> str:
        """"""
        return 'MaxPool2D'
