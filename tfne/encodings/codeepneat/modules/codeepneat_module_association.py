from .codeepneat_module_densedropout import CoDeepNEATModuleDenseDropout
from .codeepneat_module_conv2dmaxpool2d import CoDeepNEATModuleConv2DMaxPool2D
from .codeepneat_module_dropout import CoDeepNEATModuleDropout

# Dict associating the string name of the module when referenced in CoDeepNEAT config with the concrete instance of
# the respective module
MODULES = {
    'DenseDropout': CoDeepNEATModuleDenseDropout,
    'Conv2DMaxPool2D': CoDeepNEATModuleConv2DMaxPool2D,
    'Dropout': CoDeepNEATModuleDropout
}
