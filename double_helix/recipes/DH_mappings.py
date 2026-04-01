
from PYME.recipes.base import ModuleBase, register_module
from PYME.recipes.traits import Input, Output, FileOrURI, Float, Enum, Int, List, Bool
from PYME.IO import tabular
from matplotlib import image
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
    correct_wobble = Bool(True)
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
        dh_loc, rec_plot = lookup_dh_z(dh_loc, calibration, rough_knot_spacing=self.target_knot_spacing, plot=True, wobble_correction=self.correct_wobble)

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
    output_norm_factor = Output('dh_norm_factor')

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
                    # transpose frame because detector didn't originally match PYME XY convention
                    strength_image, angle_image = detector.filter_frame(frame.squeeze().T)
                    strength[:, :, z_ind, t_ind, c_ind] = strength_image.T
                    noise_sigma = fitTask.calcSigma(im.mdh, frame)
                    row, col, angle = detector.extract_candidates(strength_image, angle_image,
                                                                detector.normFactor * (self.thresh * noise_sigma.squeeze().T)**2)
                    r = np.append(r, row)
                    c = np.append(c, col)
                    theta = np.append(theta, angle)
                    zi = np.append(zi, [z_ind] * len(row))
                    ti = np.append(ti, [t_ind] * len(row))
                    ci = np.append(ci, [c_ind] * len(row))
        
        detections = DictSource({
            # NOTE - XY vs RC trickery to match PYME convention.
            'x': c*im.voxelsize_nm.y, # Would be r*im.voxelsize_nm.x, but swap here to match PYME convention
            'y': r*im.voxelsize_nm.x, # Would be c*im.voxelsize_nm.y, but swap here to match PYME convention
            'angle': theta, 'z_index': zi, 't': ti, 'probe': ci,
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

        norm_factor_table = tabular.DictSource({'norm_factor': np.atleast_1d(detector.normFactor)})
        return {'output_strength_im': strength, 'output_detections': detections, 'output_norm_factor': norm_factor_table}


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

    lobe_sep_nm = Float(1050)
    lobe_sigma_nm = Float(210)
    filter_sigma_px_stride = Float(0.25)
    fit_roi_half_size = Int(10)
    fit_module = Enum(['double_helix.DoubleGaussFit'])
    
    output_plot = Output('filter_sigma_plot')
    output_data = Output('max_strength')

    def run(self, input_image):

        import matplotlib.pyplot as plt
        from PYME.IO.image import ImageStack  # FIXME - would be nice to pass output max strength array as a simple array instead
        from PYME.recipes.measurement import FitPoints
        from scipy.signal import find_peaks

        #### Fit Z Stack
        vs_x_nm = input_image.mdh['voxelsize']['x']*1e3
        vs_y_nm = input_image.mdh['voxelsize']['y']*1e3
  
        n_steps = input_image.data_xyztc.shape[2]
        obj_positions = {}

        obj_positions['x'] = vs_x_nm * 0.5 * input_image.data_xyztc.shape[0] * np.ones(n_steps)
        obj_positions['y'] = vs_y_nm * 0.5 * input_image.data_xyztc.shape[1] * np.ones(n_steps)
        obj_positions['t'] = np.arange(input_image.data.shape[2])
        z = np.arange(input_image.data_xyztc.shape[2]) * input_image.mdh['voxelsize.z'] * 1.e3  # [um -> nm]
        obj_positions['z'] = z - z.mean()

        results = []

        detection_params = {
        'Analysis.ROISize': self.fit_roi_half_size,
        'Analysis.LobeSepGuess': self.lobe_sep_nm,
        'Analysis.SigmaGuess': self.lobe_sigma_nm
        }

        for chan_ind in range(input_image.data_xyztc.shape[3]):
            mod = FitPoints(roiHalfSize=self.fit_roi_half_size,
                            fitModule=self.fit_module, 
                            channel=chan_ind,
                            parameters=detection_params)
            res = mod.apply_simple(inputImage=input_image, inputPositions=obj_positions)
            

            results.append({
                    'theta': res['fitResults_theta'].tolist(),
                    'lobesep': res['fitResults_lobesep'].tolist(),
                    'sigma': res['fitResults_sigma'].tolist(),
                    'x': res['x'].tolist(),
                    'y': res['y'].tolist(),
                    'z': obj_positions['z'].tolist(),
                })
            

        # Plot Fitted Theta, Lobe Sep., Lobe Sigma vs Z
        plt.ioff()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for ind, res in enumerate(results):
            axes[0].plot(res['z'], res['theta'], label='chan. %d' % ind)
            axes[0].set_ylabel('Theta [rad.]')

            axes[1].plot(res['z'], res['lobesep'], label='chan. %d' % ind)
            axes[1].set_ylabel('Lobe Separation [nm]')
            median_lobesep = np.median(res['lobesep'])
            axes[1].axhline(y=median_lobesep, color='r', linestyle='--', linewidth=2, 
                         label=f'Median Lobe Separation, {median_lobesep:.0f} nm')
        
            
            axes[2].plot(res['z'], res['sigma'], label='chan. %d' % ind)
            axes[2].set_ylabel('Sigma [nm]')
            median_sigma = np.median(res['sigma'])
            axes[2].axhline(y=median_sigma, color='r', linestyle='--', linewidth=2, 
                         label=f'Median Sigma, {median_sigma:.0f} nm')

        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[2].set_xlabel('z position [nm]')

        

        plt.tight_layout()

        plt.ion()
        plt.show()
        ####    

        # Determine x and y pixel corresponding to center of DHPSF fit of each frame in z stack
        x_0_px = np.floor(results[0]['x']/vs_x_nm).astype(int)
        y_0_px = np.floor(results[0]['y']/vs_y_nm).astype(int)

        # Create array of index of theta values corresponding to theta displacement of -80, -40, 0, 40, and 80 degrees from middle z slice
        theta = np.asarray(results[0]['theta'])
        theta_ind_m = np.floor(theta.size/2).astype(int)
        delta_theta = (theta - theta[theta_ind_m])

        for ind in range(len(delta_theta)):
            if np.abs(delta_theta[ind]) > np.pi/2:
                delta_theta[ind] = delta_theta[ind] - np.pi

        theta_ind_ll = np.searchsorted(delta_theta, -80*np.pi/180)
        theta_ind_l  = np.searchsorted(delta_theta, -40*np.pi/180)
        theta_ind_u  = np.searchsorted(delta_theta,  40*np.pi/180)
        theta_ind_uu = np.searchsorted(delta_theta,  80*np.pi/180)

        theta_ind = [theta_ind_ll, theta_ind_l, theta_ind_m, theta_ind_u, theta_ind_uu]

        # Determine upper and lower bounds for filter sigma search
        # lower bound: 1.5 x lobe sigma guess
        # upper bound: lobe separation guess 
        # Then create array of filter sigma values
        lobe_sigma_px = self.lobe_sigma_nm / vs_x_nm
        lobe_sep_px = self.lobe_sep_nm / vs_x_nm

        sigma_range_min = np.ceil(1*lobe_sigma_px).astype(int)
        sigma_range_max = np.ceil(lobe_sep_px).astype(int)

        filter_sigmas = np.arange(sigma_range_min, 
                                  sigma_range_max + self.filter_sigma_px_stride, 
                                  self.filter_sigma_px_stride)
        
        # Create array to store maximum value of detected strength at each theta of interest
        max_strength = np.zeros((len(filter_sigmas), len(theta_ind)) + input_image.data_xyztc.shape[3:], dtype=float)

        # Run detection with each filter sigma on each slice
        # Save strength at each theta of interest to max_strength
        # Strength is normalized by dividing by normFactor and taking square root 
        for s_ind, sigma in enumerate(filter_sigmas):
            det = DetectDoubleHelices(lobe_sep_nm=self.lobe_sep_nm, 
                                        lobe_sigma_nm=self.lobe_sigma_nm, filter_sigma_px=sigma,
                                        fit_roi_half_size=self.fit_roi_half_size, 
                                        fit_module=self.fit_module)
            outputs = det.apply(input_image=input_image)
            strength_stack = outputs[det.output_strength_im]
            norm_factor = outputs[det.output_norm_factor]['norm_factor'][0]
            for c_ind in range(strength_stack.data_xyztc.shape[4]):
                for ang_ind in range(len(theta_ind)):
                    for t_ind in range(strength_stack.data_xyztc.shape[3]):
                        max_strength[s_ind, ang_ind, t_ind, c_ind] = np.sqrt(strength_stack.data_xyztc[x_0_px[t_ind], y_0_px[t_ind], ang_ind, t_ind, c_ind]/norm_factor)


        # plot the results
        # Average strength at each theta of interest for each filter sigma is plotted
        n_channels = strength_stack.data_xyztc.shape[4]
        fig, axes = plt.subplots(n_channels, 1, figsize=(4, 3 * n_channels), dpi=200)
        for c_ind in range(n_channels):
            try:
                ax = axes[c_ind]
            except:
                ax = axes
            # average over theta_ind (axis=1) and z/t (axis=2), leaving only filter_sigma axis
            mean_strength = max_strength[:, :, :, c_ind].mean(axis=(1, 2))  # shape: (n_sigmas,)
            ax.plot(filter_sigmas, mean_strength)
            ax.set_xlabel('Filter Sigma [px]')
            ax.set_ylabel('Mean Strength')
            peak_indices, _ = find_peaks(mean_strength)
            optimal_sigma = np.max(filter_sigmas[peak_indices])
            ax.set_title(f'Chan{c_ind}, max at Sigma = {optimal_sigma:.2f} [px]')
            ax.axvline(x=optimal_sigma, color='r', linestyle='--', label=f'Max at {optimal_sigma:.2f} px')
            ax.legend()

        plt.tight_layout()
        return {'output_plot': fig, 'output_data': ImageStack(data=max_strength, haveGUI=False)}
