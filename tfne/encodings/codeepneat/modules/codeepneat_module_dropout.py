from __future__ import annotations

import math
import random
import statistics

import numpy as np
import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase
from tfne.helper_functions import round_with_step


class CoDeepNEATModuleDropout(CoDeepNEATModuleBase):
    """
    TFNE CoDeepNEAT module encapsulating a Dropout layer.
    """

    def __init__(self,
                 config_params,
                 module_id,
                 parent_mutation,
                 dtype,
                 rate=None,
                 self_initialization_flag=False):
        """
        Create module by storing supplied parameters. If self initialization flag is supplied, randomly initialize the
        module parameters based on the range of parameters allowed by config_params
        @param config_params: dict of the module parameter range supplied via config
        @param module_id: int of unique module ID
        @param parent_mutation: dict summarizing the mutation of the parent module
        @param dtype: string of deserializable TF dtype
        @param rate: see TF documentation
        @param self_initialization_flag: bool flag indicating if all module parameters should be randomly initialized
        """
        # Register the implementation specifics by calling parent class
        super().__init__(config_params, module_id, parent_mutation, dtype)

        # Register the module parameters
        self.rate = rate

        # If self initialization flag is provided, initialize the module parameters as they are currently set to None
        if self_initialization_flag:
            self._initialize()

    def __str__(self) -> str:
        """
        @return: string representation of the module
        """
        return "CoDeepNEAT Dropout Module | ID: {:>6} | Fitness: {:>6} | Rate: {:>4}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    self.rate)

    def _initialize(self):
        """
        Randomly initialize all parameters of the module based on the range of parameters allowed by the config_params
        variable.
        """
        # Uniform randomly set module parameters
        self.rate = random.uniform(self.config_params['rate']['min'],
                                             self.config_params['rate']['max'])

    def create_module_layers(self) -> (tf.keras.layers.Layer, ...):
        """
        Instantiate TF layers with their respective configuration that are represented by the current module
        configuration. Return the instantiated module layers in their respective order as a tuple.
        @return: tuple of instantiated TF layers represented by the module configuration.
        """
        # Create iterable that contains all layers concatenated in this module
        module_layers = list()

        dropout_layer = tf.keras.layers.Dropout(rate=self.rate,
                                                    dtype=self.dtype)
        module_layers.append(dropout_layer)

        # Return the iterable containing all layers present in the module
        return module_layers

    def create_downsampling_layer(self, in_shape, out_shape) -> tf.keras.layers.Layer:
        """"""
        raise NotImplementedError("Downsampling has not yet been implemented for Dropout Modules")

    def create_mutation(self,
                        offspring_id,
                        max_degree_of_mutation) -> CoDeepNEATModuleDropout:
        """
        Create mutated Dropout module and return it. Categorical parameters are chosen randomly from all
        available values. Sortable parameters are perturbed through a random normal distribution with the current value
        as mean and the config specified stddev
        @param offspring_id: int of unique module ID of the offspring
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated Dropout module with mutated parameters
        """
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'rate': self.rate}

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 1)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(1), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                perturbed_rate = np.random.normal(loc=self.rate,
                                                    scale=self.config_params['rate']['stddev'])
                offspring_params['rate'] = max(min(perturbed_rate, self.config_params['rate']['max']), self.config_params['rate']['min'])
                parent_mutation['mutated_params']['rate'] = self.rate

        return CoDeepNEATModuleDropout(config_params=self.config_params,
                                                      module_id=offspring_id,
                                                      parent_mutation=parent_mutation,
                                                      dtype=self.dtype,
                                                      **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> CoDeepNEATModuleDropout:
        """
        Create crossed over Dropout module and return it. Carry over parameters of fitter parent for
        categorical parameters and calculate parameter average between both modules for sortable parameters
        @param offspring_id: int of unique module ID of the offspring
        @param less_fit_module: second Dropout module with lower fitness
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated Dropout module with crossed over parameters
        """
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        crossed_over_rate = ((self.rate + less_fit_module.rate) / 2)
        offspring_params['rate'] = crossed_over_rate

        return CoDeepNEATModuleDropout(config_params=self.config_params,
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
            'rate': self.rate
        }

    def get_distance(self, other_module) -> float:
        """
        Calculate distance between 2 Dropout modules by inspecting each parameter, calculating the
        congruence between each and eventually averaging the out the congruence. The distance is returned as the average
        congruences distance to 1.0. The congruence of continuous parameters is calculated by their relative distance.
        The congruence of categorical parameters is either 1.0 in case they are the same or it's 1 divided to the amount
        of possible values for that specific parameter. Return the calculated distance.
        @param other_module: second Dropout module to which the distance has to be calculated
        @return: float between 0 and 1. High values indicating difference, low values indicating similarity
        """

        # Return the distance as the distance of the average congruence to the perfect congruence of 1.0
        return round(1.0 - (self.rate - other_module.rate)**2, 4)

    def get_module_type(self) -> str:
        """"""
        return 'Dropout'
