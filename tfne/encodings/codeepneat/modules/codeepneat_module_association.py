from .codeepneat_module_densedropout import CoDeepNEATModuleDenseDropout
from .codeepneat_module_conv2d import CoDeepNEATModuleConv2D
from .codeepneat_module_maxpool2d import CoDeepNEATModuleMaxPool2D
from .codeepneat_module_dropout import CoDeepNEATModuleDropout
from .codeepneat_module_activation import CoDeepNEATModuleActivation

# Dict associating the string name of the module when referenced in CoDeepNEAT config with the concrete instance of
# the respective module
MODULES = {
    'DenseDropout': CoDeepNEATModuleDenseDropout,
    'Conv2D': CoDeepNEATModuleConv2D,
    'Dropout': CoDeepNEATModuleDropout,
    'MaxPool2D': CoDeepNEATModuleMaxPool2D,
    'Activation': CoDeepNEATModuleActivation
}
