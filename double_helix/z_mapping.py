
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

def calibrate_double_helix_psf(image, fit_module, roi_half_size=11):
    """Generate Z vs theta calibration information from an PSF image stack

    Parameters
    ----------
    image : PYME.IO.image.ImageStack
       image of single, extracted PSF, with even z spacing between slices. If
       one is working with imagestacks at uneven Z or with multiple frames per
       step, one should first average by z step, and then resample the PSF stack
    fit_module : str
        FitFactory to use when fitting each frame. At the moment, only supports
        DoubleHelix_Theta
    roi_half_size : int, optional
        half size of fitting ROI, by default 11 which results in 11 * 2 + 1 = 23
        pixel ROI.

    Returns
    -------
    results
        list of double helix calibration dictionaries, one per color channel.
    """
    from PYME.recipes.measurement import FitPoints
    # At the moment, this z spacing to be even along the PSF.
    # one should first average by z, and then resample PSF stack, then finally
    # run this calibration. 
    vs_x_nm = image.voxelsize_nm.x
    vs_y_nm = image.voxelsize_nm.y
    # z_step_nm = image.voxelsize_nm.z
    n_steps = image.data_xyztc.shape[2]
    obj_positions = {}

    obj_positions['x'] = vs_x_nm * 0.5 * image.data_xyztc.shape[0] * np.ones(n_steps)
    obj_positions['y'] = vs_y_nm * 0.5 * image.data_xyztc.shape[1] * np.ones(n_steps)
    obj_positions['t'] = np.arange(image.data.shape[2])
    z = np.arange(image.data_xyztc.shape[2]) * image.mdh['voxelsize.z'] * 1.e3  # [um -> nm]
    obj_positions['z'] = z - z.mean()

    results = []

    for chan_ind in range(image.data_xyztc.shape[3]):
        res = FitPoints(roiHalfSize=roi_half_size,
                        fitModule=fit_module, 
                        channel=chan_ind).apply_simple(
                                            inputImage=image, 
                                            inputPositions=obj_positions,
                                            )
        
        results.append({
                'theta': res['fitResults_theta'].tolist(),
                'lobesep': res['fitResults_lobesep'].tolist(),
                'sigma': res['fitResults_sigma'].tolist(),
                'z': obj_positions['z'].tolist(),
            })

    return results

def lookup_dh_z(fres, calibration, rough_knot_spacing=101., plot=False):
    """
    Generates a look-up table for z based on theta fit results and calibration 
    information.

    Parameters
    ----------
    fres : dict-like
        Contains fit results (localizations) to be mapped in z
    calibration : list
        Each element is a dictionary corresponding to a color channel, and 
        contains the fitted double helix orientation (theta [rad.]) for 
        various z-positions
    rough_knot_spacing : Float
        Smoothing is applied to the theta look-up curves by fitting a cubic 
        spline with knots spaced roughly at intervals of rough_knot_spacing (in 
        nanometers). There is potentially rounding within the step-size of the
        calibration to make knot placing more convenient.
    plot : bool
        Flag to toggle plotting

    Returns
    -------
    dh_loc : PYME.IO.tabular.MappingFilter
        fit reuslts with 'dz_z' and 'dh_z_lookup_error' keys for double-helix
        z [nm] and associated uncertianty [nm]
    dh_z_lookup_plot : PYME.recipes.graphic.Plot
        PYME recipe plot object for viewing / saving graph of used calibration
        with resulting z localizations overlayed.

    """
    from PYME.IO.tabular import ColourFilter, MappingFilter, ConcatenateFilter
    from PYME.recipes.graphing import Plot

    # find overall min and max z values
    z_min = 0
    z_max = 0
    for cal in calibration: #more idiomatic way of looping through list - also avoids one list access / lookup
        r_min, r_max = cal['z_range']
        z_min = min(z_min, float(r_min))
        z_max = max(z_max, float(r_max))

    # generate z vector for interpolation
    z_v = np.arange(z_min, z_max)
    # store for plotting later
    theta_cals = []
    # sig_cals = []
    # lobesep_cals = []

    for c_ind, cal in enumerate(calibration):
        # grab localizations corresponding to this channel
        chan = ColourFilter(fres, currentColour=c_ind)
        chan = MappingFilter(chan)
        # sigma = chan['fitResults_sigma']
        # error_sigma = chan['fitError_sigma']
        theta = chan['fitResults_theta']
        error_theta = chan['fitError_theta']
        # lobesep = chan['fitResults_lobesep']
        # error_lobesep = chan['fitError_lobesep']


        zdat = np.array(cal['z'])
        # grab indices of range we trust
        z_range = cal['z_range']
        z_valid_mask = (zdat > float(z_range[0]))*(zdat < float(z_range[1]))
        z_valid = zdat[z_valid_mask]
        # generate splines with knots spaced roughly as rough_knot_spacing [nm]
        z_steps = np.unique(z_valid)
        dz_med = np.median(np.diff(z_steps))
        smoothing_factor = int(rough_knot_spacing / (dz_med))
        knots = z_steps[1:-1:smoothing_factor]

        # sig_cal = LSQUnivariateSpline(z_valid, np.array(cal['sigma'])[z_valid_mask], knots, ext='const')(z_v)
        # sig_cals.append(sig_cal)
        # make sure we don't have a pi jump in the middle of our spline!
        unwrapped_theta_cal = np.unwrap(np.asarray(cal['theta'])[z_valid_mask], np.pi/2, period=np.pi)
        theta_cal = LSQUnivariateSpline(z_valid, unwrapped_theta_cal, knots, ext='const')(z_v)
        theta_cals.append(theta_cal)
        # lobesep_cal = LSQUnivariateSpline(z_valid, np.array(cal['lobesep'])[z_valid_mask], knots, ext='const')(z_v)
        # lobesep_cals.append(lobesep_cal)

        z_out = np.empty_like(theta)
        error_z_out = np.empty_like(theta)
        # loop over each localization and find it's Z position
        for ind in range(len(theta)):
            # sigma_residual = np.abs(sigma[ind] - sig_cal)  # [nm]

            # use sin in the residual calc to handle wrapping
            theta_residual = np.sin(theta[ind] - theta_cal)  # [rad., under small angle approximation]
            # lobesep_residual = lobesep[ind] - lobesep_cal  # [nm]
            
            theta_normed = theta_residual ** 2
            # lobesep_normed = lobesep_residual ** 2

            # min_loc = np.argmin(np.stack([
            #     # sigma_normed, 
            #     theta_normed, 
            #     lobesep_normed
            # ], axis=0).sum(axis=0))
            min_loc = np.argmin(theta_normed)

            z_out[ind] = z_v[min_loc]
            error_z_out[ind] = 1  # FIXME
        
        chan.addColumn('dh_z', z_out)
        chan.addColumn('dh_z_lookup_error', error_z_out)            

        if c_ind < 1:
            dh_loc = chan
        else:
            dh_loc = ConcatenateFilter(dh_loc, chan)
    dh_loc.setMapping('z', 'dh_z + z')
    
    if not plot:
        return dh_loc
    else:
        return dh_loc, Plot(lambda: plot_dh_z_lookup([{
            'theta': theta_cals[c_ind],
            # 'sigma': sig_cals[c_ind],
            # 'lobesep': lobesep_cals[c_ind],
            'z_v': z_v
        } for c_ind in range(len(calibration))], dh_loc))

def plot_dh_z_lookup(calibration_splines, dh_loc):
    from matplotlib import pyplot as plt
    from PYME.IO.tabular import ColourFilter
    fig = plt.figure()
    # plt.subplots(2, 1)

    for c_ind, cal in enumerate(calibration_splines):
        # grab localizations corresponding to this channel
        chan = ColourFilter(dh_loc, currentColour=c_ind)

        # sigma = chan['fitResults_sigma']
        # error_sigma = chan['fitError_sigma']
        theta = chan['fitResults_theta']
        error_theta = chan['fitError_theta']
        # lobesep = chan['fitResults_lobesep']
        # error_lobesep = chan['fitError_lobesep']
        z_out = chan['dh_z']
        z_v = cal['z_v']

        # plt.subplot(211)
        # plt.figure()
        plt.plot(z_v, cal['theta'], ':', label='Splined Cal.')
        plt.errorbar(z_out, theta, error_theta, linestyle='')

        # plt.subplot(312)
        # plt.plot(z_v, cal['sigma'], ':', label='Splined Cal.')
        # plt.errorbar(z_out, sigma, error_sigma, linestyle='')

        # plt.subplot(212)
        # plt.plot(z_v, cal['lobesep'], ':', label='Splined Cal.')
        # plt.errorbar(z_out, lobesep, error_lobesep, linestyle='')
    
    # plt.subplot(211)
    plt.ylabel('Theta [rad]')
    plt.xlabel('Z [nm]')
    plt.legend()

    # plt.subplot(312)
    # plt.ylabel('Sigma [nm]')
    # plt.xlabel('Z [nm]')
    # plt.legend()
    
    # plt.subplot(212)
    # plt.ylabel('Lobe Separation [nm]')
    # plt.xlabel('Z [nm]')
    # plt.legend()

    return fig
