
from PYME.recipes.base import ModuleBase, register_module, Input, Output
from PYME.IO import tabular
import numpy as np

# @register_module('DoubleHelixMappings')
# class DHMappings(ModuleBase):
#     """Create a new mapping object which derives mapped keys from original ones"""
#     input_name = Input('localizations')
#     output_name = Output('dh_localizations')

#     def run(self, input_name):
#         dh_loc = tabular.MappingFilter(input_name)
#         # shorten names for convenience
#         x0 = dh_loc['fitResults_x0']
#         x1 = dh_loc['fitResults_x1']
#         y0 = dh_loc['fitResults_y0']
#         y1 = dh_loc['fitResults_y1']
#         x0_err = dh_loc['fitError_x0']
#         x1_err = dh_loc['fitError_x1']
#         y0_err = dh_loc['fitError_y0']
#         y1_err = dh_loc['fitError_y1']

#         lobe_separation = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
#         # lobe_separation_err = 1

#         x_com = 0.5 * (x0 + x1)
#         y_com = 0.5 * (y0 + y1)
#         theta = np.arctan2(x1 - x0, y1 - y0)

#         # FIXME - A and B are amplitudes, not sum-norms
#         n_adu = (dh_loc['fitResults_A'] + dh_loc['fitResults_B'])  # [ADU]
#         # data was fitted in offset + flat corrected ADU, change to e-
#         n_photoelectrons = n_photoelectrons * dh_loc.mdh['Camera.ElectronsPerCount'] /  # [e-]
        
#         dh_loc.addColumn('x', x_com)
#         dh_loc.addColumn('y', y_com)
#         dh_loc.addColumn('theta', theta)
#         dh_loc.addColumn('lobe_separation', lobe_separation)
#         dh_loc.addColumn('n_photoelectrons', n_photoelectrons)
#         return dh_loc

@register_module('DoubleHelixMapZ')
class DoubleHelixMapZ(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    input_name = Input('localizations')
    output_name = Output('dh_localizations')

    def run(self, input_name):
        dh_loc = tabular.MappingFilter(input_name)
        # shorten names for convenience
        x0 = dh_loc['fitResults_x0']
        x1 = dh_loc['fitResults_x1']
        y0 = dh_loc['fitResults_y0']
        y1 = dh_loc['fitResults_y1']
        x0_err = dh_loc['fitError_x0']
        x1_err = dh_loc['fitError_x1']
        y0_err = dh_loc['fitError_y0']
        y1_err = dh_loc['fitError_y1']

        lobe_separation = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        # lobe_separation_err = 1

        x_com = 0.5 * (x0 + x1)
        y_com = 0.5 * (y0 + y1)
        theta = np.arctan2(x1 - x0, y1 - y0)

        # FIXME - A and B are amplitudes, not sum-norms
        n_adu = (dh_loc['fitResults_A'] + dh_loc['fitResults_B'])  # [ADU]
        # data was fitted in offset + flat corrected ADU, change to e-
        n_photoelectrons = n_photoelectrons * dh_loc.mdh['Camera.ElectronsPerCount'] /  # [e-]
        
        dh_loc.addColumn('x', x_com)
        dh_loc.addColumn('y', y_com)
        dh_loc.addColumn('theta', theta)
        dh_loc.addColumn('lobe_separation', lobe_separation)
        dh_loc.addColumn('n_photoelectrons', n_photoelectrons)
        return dh_loc
