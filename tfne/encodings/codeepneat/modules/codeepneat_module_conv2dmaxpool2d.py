from __future__ import annotations

import math
import random
import statistics

import numpy as np
import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase
from tfne.helper_functions import round_with_step


class CoDeepNEATModuleConv2DMaxPool2D(CoDeepNEATModuleBase):
    """
    TFNE CoDeepNEAT module encapsulating a Conv2D layer, and a optionally following MaxPooling2D layer. The downsampling layer is another Conv2D layer.
    """

    def __init__(self,
                 config_params,
                 module_id,
                 parent_mutation,
                 dtype,
                 merge_method=None,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 activation=None,
                 kernel_init=None,
                 bias_init=None,
                 max_pool_flag=None,
                 max_pool_size=None,
                 self_initialization_flag=False):
        """
        Create module by storing supplied parameters. If self initialization flag is supplied, randomly initialize the
        module parameters based on the range of parameters allowed by config_params
        @param config_params: dict of the module parameter range supplied via config
        @param module_id: int of unique module ID
        @param parent_mutation: dict summarizing the mutation of the parent module
        @param dtype: string of deserializable TF dtype
        @param merge_method: dict representing a TF deserializable merge layer
        @param filters: see TF documentation
        @param kernel_size: see TF documentation
        @param strides: see TF documentation
        @param padding: see TF documentation
        @param activation: see TF documentation
        @param kernel_init: see TF documentation
        @param bias_init: see TF documentation
        @param max_pool_flag: see TF documentation
        @param max_pool_size: see TF documentation
        @param self_initialization_flag: bool flag indicating if all module parameters should be randomly initialized
        """
        # Register the implementation specifics by calling parent class
        super().__init__(config_params, module_id, parent_mutation, dtype)

        # Register the module parameters
        self.merge_method = merge_method
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.max_pool_flag = max_pool_flag
        self.max_pool_size = max_pool_size

        # If self initialization flag is provided, initialize the module parameters as they are currently set to None
        if self_initialization_flag:
            self._initialize()

    def __str__(self) -> str:
        """
        @return: string representation of the module
        """
        return "CoDeepNEAT Conv2D MaxPool Module | ID: {:>6} | Fitness: {:>6} | Filters: {:>4} | " \
               "Kernel: {:>6} | Activ: {:>6} | Pool Size: {:>6}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    self.filters,
                    str(self.kernel_size),
                    self.activation,
                    "None" if self.max_pool_flag is False else str(self.max_pool_size))

    def _initialize(self):
        """
        Randomly initialize all parameters of the module based on the range of parameters allowed by the config_params
        variable.
        """
        # Uniform randomly set module parameters
        self.merge_method = random.choice(self.config_params['merge_method'])
        self.merge_method['config']['dtype'] = self.dtype
        random_filters = random.randint(self.config_params['filters']['min'],
                                        self.config_params['filters']['max'])
        self.filters = round_with_step(random_filters,
                                       self.config_params['filters']['min'],
                                       self.config_params['filters']['max'],
                                       self.config_params['filters']['step'])
        self.kernel_size = random.choice(self.config_params['kernel_size'])
        self.strides = random.choice(self.config_params['strides'])
        self.padding = random.choice(self.config_params['padding'])
        self.activation = random.choice(self.config_params['activation'])
        self.kernel_init = random.choice(self.config_params['kernel_init'])
        self.bias_init = random.choice(self.config_params['bias_init'])
        self.max_pool_flag = random.random() < self.config_params['max_pool_flag']
        self.max_pool_size = random.choice(self.config_params['max_pool_size'])

    def create_module_layers(self) -> (tf.keras.layers.Layer, ...):
        """
        Instantiate TF layers with their respective configuration that are represented by the current module
        configuration. Return the instantiated module layers in their respective order as a tuple.
        @return: tuple of instantiated TF layers represented by the module configuration.
        """
        # Create iterable that contains all layers concatenated in this module
        module_layers = list()

        # Create the basic keras Convolutional 2D layer, needed in all variants of the module
        conv_layer = tf.keras.layers.Conv2D(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            activation=self.activation,
                                            kernel_initializer=self.kernel_init,
                                            bias_initializer=self.bias_init,
                                            dtype=self.dtype)
        module_layers.append(conv_layer)

        # If max pooling flag present, add the max pooling layer as configured to the module layers iterable
        if self.max_pool_flag:
            max_pool_layer = tf.keras.layers.MaxPool2D(pool_size=self.max_pool_size,
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
        # As the Conv2DMaxPool2D module downsamples with a Conv2D layer, assure that the input and output shape
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
                        max_degree_of_mutation) -> CoDeepNEATModuleConv2DMaxPool2D:
        """
        Create mutated Conv2DMaxPool2D module and return it. Categorical parameters are chosen randomly from all
        available values. Sortable parameters are perturbed through a random normal distribution with the current value
        as mean and the config specified stddev
        @param offspring_id: int of unique module ID of the offspring
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated Conv2DMaxPool2D module with mutated parameters
        """
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'filters': self.filters,
                            'kernel_size': self.kernel_size,
                            'strides': self.strides,
                            'padding': self.padding,
                            'activation': self.activation,
                            'kernel_init': self.kernel_init,
                            'bias_init': self.bias_init,
                            'max_pool_flag': self.max_pool_flag,
                            'max_pool_size': self.max_pool_size}

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 12)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(12), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                offspring_params['merge_method'] = random.choice(self.config_params['merge_method'])
                parent_mutation['mutated_params']['merge_method'] = self.merge_method
            elif param_to_mutate == 1:
                perturbed_filters = int(np.random.normal(loc=self.filters,
                                                         scale=self.config_params['filters']['stddev']))
                offspring_params['filters'] = round_with_step(perturbed_filters,
                                                              self.config_params['filters']['min'],
                                                              self.config_params['filters']['max'],
                                                              self.config_params['filters']['step'])
                parent_mutation['mutated_params']['filters'] = self.filters
            elif param_to_mutate == 2:
                offspring_params['kernel_size'] = random.choice(self.config_params['kernel_size'])
                parent_mutation['mutated_params']['kernel_size'] = self.kernel_size
            elif param_to_mutate == 3:
                offspring_params['strides'] = random.choice(self.config_params['strides'])
                parent_mutation['mutated_params']['strides'] = self.strides
            elif param_to_mutate == 4:
                offspring_params['padding'] = random.choice(self.config_params['padding'])
                parent_mutation['mutated_params']['padding'] = self.padding
            elif param_to_mutate == 5:
                offspring_params['activation'] = random.choice(self.config_params['activation'])
                parent_mutation['mutated_params']['activation'] = self.activation
            elif param_to_mutate == 6:
                offspring_params['kernel_init'] = random.choice(self.config_params['kernel_init'])
                parent_mutation['mutated_params']['kernel_init'] = self.kernel_init
            elif param_to_mutate == 7:
                offspring_params['bias_init'] = random.choice(self.config_params['bias_init'])
                parent_mutation['mutated_params']['bias_init'] = self.bias_init
            elif param_to_mutate == 8:
                offspring_params['max_pool_flag'] = not self.max_pool_flag
                parent_mutation['mutated_params']['max_pool_flag'] = self.max_pool_flag
            elif param_to_mutate == 9:
                offspring_params['max_pool_size'] = random.choice(self.config_params['max_pool_size'])
                parent_mutation['mutated_params']['max_pool_size'] = self.max_pool_size

        return CoDeepNEATModuleConv2DMaxPool2D(config_params=self.config_params,
                                                      module_id=offspring_id,
                                                      parent_mutation=parent_mutation,
                                                      dtype=self.dtype,
                                                      **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> CoDeepNEATModuleConv2DMaxPool2D:
        """
        Create crossed over Conv2DMaxPool2D module and return it. Carry over parameters of fitter parent for
        categorical parameters and calculate parameter average between both modules for sortable parameters
        @param offspring_id: int of unique module ID of the offspring
        @param less_fit_module: second Conv2DMaxPool2D module with lower fitness
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated Conv2DMaxPool2D module with crossed over parameters
        """
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        offspring_params['merge_method'] = self.merge_method
        offspring_params['filters'] = round_with_step(int((self.filters + less_fit_module.filters) / 2),
                                                      self.config_params['filters']['min'],
                                                      self.config_params['filters']['max'],
                                                      self.config_params['filters']['step'])
        offspring_params['kernel_size'] = self.kernel_size
        offspring_params['strides'] = self.strides
        offspring_params['padding'] = self.padding
        offspring_params['activation'] = self.activation
        offspring_params['kernel_init'] = self.kernel_init
        offspring_params['bias_init'] = self.bias_init
        offspring_params['max_pool_flag'] = self.max_pool_flag
        offspring_params['max_pool_size'] = self.max_pool_size

        return CoDeepNEATModuleConv2DMaxPool2D(config_params=self.config_params,
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
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'kernel_init': self.kernel_init,
            'bias_init': self.bias_init,
            'max_pool_flag': self.max_pool_flag,
            'max_pool_size': self.max_pool_size
        }

    def get_distance(self, other_module) -> float:
        """
        Calculate distance between 2 Conv2DMaxPool2D modules by inspecting each parameter, calculating the
        congruence between each and eventually averaging the out the congruence. The distance is returned as the average
        congruences distance to 1.0. The congruence of continuous parameters is calculated by their relative distance.
        The congruence of categorical parameters is either 1.0 in case they are the same or it's 1 divided to the amount
        of possible values for that specific parameter. Return the calculated distance.
        @param other_module: second Conv2DMaxPool2D module to which the distance has to be calculated
        @return: float between 0 and 1. High values indicating difference, low values indicating similarity
        """
        if not isinstance(other_module, CoDeepNEATModuleConv2DMaxPool2D):
            return 0.0
    
        congruence_list = list()
        if self.merge_method == other_module.merge_method:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['merge_method']))
        if self.filters >= other_module.filters:
            congruence_list.append(other_module.filters / self.filters)
        else:
            congruence_list.append(self.filters / other_module.filters)
        if self.kernel_size == other_module.kernel_size:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['kernel_size']))
        if self.strides == other_module.strides:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['strides']))
        if self.padding == other_module.padding:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['padding']))
        if self.activation == other_module.activation:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['activation']))
        if self.kernel_init == other_module.kernel_init:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['kernel_init']))
        if self.bias_init == other_module.bias_init:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['bias_init']))
        congruence_list.append(abs(self.max_pool_flag - other_module.max_pool_flag))
        if self.max_pool_size == other_module.max_pool_size:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['max_pool_size']))

        # Return the distance as the distance of the average congruence to the perfect congruence of 1.0
        return round(1.0 - statistics.mean(congruence_list), 4)

    def get_module_type(self) -> str:
        """"""
        return 'Conv2DMaxPool2D'
