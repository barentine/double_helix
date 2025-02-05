
from PYME.recipes.base import ModuleBase, register_module
from PYME.recipes.traits import Input, Output, FileOrURI, Float, Enum, Int, List
from PYME.IO import tabular
import numpy as np
import logging
logger = logging.getLogger(__name__)

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
    calibration_location = FileOrURI('')
    target_knot_spacing = Float(101)
    plot_name = Output('dh_z_lookup_plot')
    output_name = Output('dh_localizations')

    def run(self, input_name):
        from double_helix.z_mapping import lookup_dh_z
        from PYME.IO import unifiedIO, MetaDataHandler
        import ujson as json
        
        dh_loc = tabular.MappingFilter(input_name)

        s = unifiedIO.read(self.calibration_location)
        calibration = json.loads(s)

        # dh_loc = lookup_dh_z(dh_loc, calibration, rough_knot_spacing=self.target_knot_spacing)
        dh_loc, rec_plot = lookup_dh_z(dh_loc, calibration, rough_knot_spacing=self.target_knot_spacing, plot=True)

        # FIXME - A and B are amplitudes, not sum-norms
        # n_adu = (dh_loc['fitResults_A'] + dh_loc['fitResults_B'])  # [ADU]
        # # data was fitted in offset + flat corrected ADU, change to e-
        # n_photoelectrons = n_photoelectrons * dh_loc.mdh['Camera.ElectronsPerCount'] /  # [e-]
        
        # dh_loc.addColumn('dh_z', z)
        # dh_loc.addColumn('dh_z_lookup_error', zerr)
        # dh_loc.setMapping('z', 'dh_z + z')

        dh_loc.mdh = MetaDataHandler.NestedClassMDHandler(input_name.mdh)
        dh_loc.mdh['Analysis.dh_calibration_used'] = self.calibration_location
        # return dh_loc
        return {
            'output_name': dh_loc,
            'plot_name': rec_plot
        }

@register_module('DetectDoubleHelices')
class DetectDoubleHelices(ModuleBase):
    """Run double helix detection on an imagestack
    
    Parameters:
    -----------
    input_image: ImageStack
        The image stack to detect double helices on
    lobe_sep_nm: float
        Initial guess for lobe separation in nm
    lobe_sigma_nm: float
        Initial guess for lobe sigma in nm
    filter_sigma_px: float
        Sigma of the filter to use in px
    fit_roi_half_size: int
        Half size of the ROI to use for fitting
    fit_module: str
        Selects which fitter to use. Fully-resolved, such that it can be
        imported, e.g. 'double_helix.DoubleGaussFit'
    thresh: float
        Threshold for detection strength.This scalar is first multiplied by the SNR estimate at 
        each pixel before the threshold is applied
    
    Notes
    -----

    Input image series should already be camera corrected (see PYME.recipes.Processing.FlatfieldAndDarkCorrect)

    Background subtraction is not currently supported by this module.
    
    """
    input_image = Input('input')
    lobe_sep_nm = Float(900)
    lobe_sigma_nm = Float(180)
    filter_sigma_px = Float(8.0)
    fit_roi_half_size = Int(10)
    fit_module = Enum(['double_helix.DoubleGaussFit'])
    thresh = Float(1.0)
    output_strength_im = Output('dh_filtered')
    output_detections = Output('dh_detections')

    def run(self, input_image):
        from PYME.IO.tabular import DictSource
        from PYME.IO.MetaDataHandler import NestedClassMDHandler
        from PYME.IO.image import ImageStack
        from PYME.localization.remFitBuf import fitTask
        import numpy as np
        import importlib
        fit_module = importlib.import_module(self.fit_module)
        
        im = input_image
        
        detector = fit_module.Detector(roi_half_size=self.fit_roi_half_size, l_initial=self.lobe_sep_nm,
                                       lobe_sigma_initial=self.lobe_sigma_nm, filter_sigma=self.filter_sigma_px,
                                       px_size_nm=im.voxelsize_nm.x)
        logger.debug(f'Filtering image with sigma={detector.filter_sigma} px')
        strength = np.zeros(im.data_xyztc.shape, dtype=float)

        r, c, = np.array([], dtype=int), np.array([], dtype=int), 
        theta = np.array([], dtype=float), 
        zi, ti, ci = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

        # filter the image
        for c_ind in range(im.data_xyztc.shape[4]):
            for z_ind in range(im.data_xyztc.shape[2]):
                for t_ind in range(im.data_xyztc.shape[3]):
                    frame = im.data_xyztc[:, :, z_ind, t_ind, c_ind]
                    strength_image, angle_image = detector.filter_frame(frame.squeeze())
                    strength[:, :, z_ind, t_ind, c_ind] = strength_image
                    noise_sigma = fitTask.calcSigma(im.mdh, frame)
                    row, col, angle = detector.extract_candidates(strength_image, angle_image, self.thresh * noise_sigma.squeeze())
                    r = np.append(r, row)
                    c = np.append(c, col)
                    theta = np.append(theta, angle)
                    zi = np.append(zi, [z_ind] * len(row))
                    ti = np.append(ti, [t_ind] * len(row))
                    ci = np.append(ci, [c_ind] * len(row))
        
        detections = DictSource({
            'x': r*im.voxelsize_nm.x, 'y': c*im.voxelsize_nm.y, 'angle': theta, 'z_index': zi, 't': ti, 'probe': ci,
            'z': zi * im.voxelsize_nm.z
        })
        detections.mdh = NestedClassMDHandler(im.mdh)
        # detections.mdh['Analysis.dh_detector'] = self.fit_module
        detections.mdh['Analysis.FilterSigma'] = self.filter_sigma_px
        detections.mdh['Analysis.ROISize'] = self.fit_roi_half_size
        detections.mdh['Analysis.LobeSepGuess'] = self.lobe_sep_nm
        detections.mdh['Analysis.SigmaGuess'] = self.lobe_sigma_nm

        stregnth_mdh = NestedClassMDHandler(detections.mdh)
        logger.debug('Strength image size:' + str(strength.shape))
        strength = ImageStack(data=strength, mdh=stregnth_mdh, haveGUI=False)

        return {'output_strength_im': strength, 'output_detections': detections}


@register_module('OptimizeFilterSigma')
class OptimizeFilterSigma(ModuleBase):
    """Run a sweep over filter sigmas along an entire image stack (typically
    an extracted PSF z stack) to determine the optimal filter sigma for detection.

    Parameters
    ----------
    input_image: ImageStack
        The image stack to detect double helices on
    lobe_sep_nm: float
        Initial guess for lobe separation in nm
    lobe_sigma_nm: float
        Initial guess for lobe sigma in nm
    filter_sigma_px_range: list of float
        Lower and upper bounds for sigma of the filter to use in px
    filter_sigma_px_stride: float
        Stride for the filter sigma sweep
    fit_roi_half_size: int
        Half size of the ROI to use for fitting
    fit_module: str
        Selects which fitter to use. Fully-resolved, such that it can be
        imported, e.g. 'double_helix.DoubleGaussFit'

    Returns
    -------
    output_plot: 
        matplotlib.pyplot figure handle showing the maximum strength resulting from each filter sigma
        on each frame of the input image stack
    output_data:
        ImageStack holding the maximum strength vs. filter sigma and frame.
    
    Notes
    -----
    Supports multiple color channels, but only a Z stack OR a Time series, not both at the same time.

    """
    input_image = Input('input')

    lobe_sep_nm = Float(900)
    lobe_sigma_nm = Float(180)
    filter_sigma_px_range = List(Float, value=[3, 15], minlen=2, maxlen=2)
    filter_sigma_px_stride = Float(0.5)
    fit_roi_half_size = Int(10)
    fit_module = Enum(['double_helix.DoubleGaussFit'])
    
    output_plot = Output('filter_sigma_plot')
    output_data = Output('max_strength')

    def run(self, input_image):
        import matplotlib.pyplot as plt
        from PYME.IO.image import ImageStack  # FIXME - would be nice to pass output max strength array as a simple array instead
        filter_sigmas = np.arange(self.filter_sigma_px_range[0], 
                                  self.filter_sigma_px_range[1] + self.filter_sigma_px_stride, 
                                  self.filter_sigma_px_stride)
        
        max_strength = np.zeros((len(filter_sigmas),) + input_image.data_xyztc.shape[2:], dtype=float)

        for s_ind, sigma in enumerate(filter_sigmas):
            det = DetectDoubleHelices(lobe_sep_nm=self.lobe_sep_nm, 
                                        lobe_sigma_nm=self.lobe_sigma_nm, filter_sigma_px=sigma,
                                        fit_roi_half_size=self.fit_roi_half_size, 
                                        fit_module=self.fit_module)
            outputs = det.apply(input_image=input_image)
            strength_stack = outputs[det.output_strength_im] 
            for c_ind in range(strength_stack.data_xyztc.shape[4]):
                for z_ind in range(strength_stack.data_xyztc.shape[2]):
                    for t_ind in range(strength_stack.data_xyztc.shape[3]):
                        max_strength[s_ind, z_ind, t_ind, c_ind] = np.max(strength_stack.data_xyztc[:, :, z_ind, t_ind, c_ind])
        
        
        # plot the results
        n_channels = strength_stack.data_xyztc.shape[4]
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 10))
        for c_ind in range(n_channels):
            try:
                ax = axes[c_ind]
            except:
                ax = axes
            to_plot = max_strength[:, :, :, c_ind].squeeze().T  # put filter sigma on the horizontal x axis
            if input_image.data_xyztc.shape[2] == 1: # long axis is t not z
                yaxis = range(to_plot.shape[0])
                yaxis_label = 'Frame'
            else:# long axis is z not t
                yaxis = input_image.voxelsize_nm.z * range(to_plot.shape[0])
                yaxis_label = 'Z [nm]'
            yaxis_px_stride = yaxis[1] - yaxis[0]
            filter_sigma_px_stride = filter_sigmas[1] - filter_sigmas[0]
            pim = ax.imshow(to_plot, interpolation='nearest', origin='lower',
                      extent=(filter_sigmas[0] - 0.5 * filter_sigma_px_stride, 
                              filter_sigmas[-1] + 0.5 * filter_sigma_px_stride, 
                              yaxis[0] - 0.5*yaxis_px_stride, yaxis[-1] + 0.5*yaxis_px_stride),
                      aspect='auto')
            plt.colorbar(pim, ax=ax)
            ax.set_xlabel('Filter Sigma [px]')
            ax.set_ylabel(yaxis_label)
            max_sigma = filter_sigmas[np.argmax(to_plot.max(axis=0))]
            ax.set_title(f'Max Strength for Chan{c_ind} at Sigma = {max_sigma} [px]')
        
        return {'output_plot': fig, 'output_data': ImageStack(data=max_strength, haveGUI=False)}
