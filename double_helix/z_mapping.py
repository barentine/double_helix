
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import LSQUnivariateSpline
import math

def calibrate_double_helix_psf(image, fit_module, roi_half_size=11, filter_sigma=5.0, lobe_sep_guess=1000, lobe_sigma_guess=200):
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
        half size of fitting ROI and half size of convolution kernels, by default 11 which results in 11 * 2 + 1 = 23
        pixel ROI.
    filter_sigma : float, optional
        sigma in pixels of G2, H2 filter functions, by default 5.0
    LobseSepGuess : float, optional
        Initial guess for the lobe separation in nm, by default 1000
    SigmaGuess : float, optional
        Initial guess for the lobe sigma in nm, by default 200

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

    # save detector parameters as dictionary that will be added to metadata used for FromPoint fitting
    detection_params = {
        'Analysis.ROISize': roi_half_size,
        'Analysis.DetectionFilterSigma': filter_sigma,
        'Analysis.LobeSepGuess': lobe_sep_guess,
        'Analysis.SigmaGuess': lobe_sigma_guess
    }
    for chan_ind in range(image.data_xyztc.shape[3]):
        mod = FitPoints(roiHalfSize=roi_half_size,
                        fitModule=fit_module, 
                        channel=chan_ind,
                        parameters=detection_params)
        res = mod.apply_simple(inputImage=image, inputPositions=obj_positions)


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
    sig_cals = []
    lobesep_cals = []

    for c_ind, cal in enumerate(calibration):
        # grab localizations corresponding to this channel
        chan = ColourFilter(fres, currentColour=c_ind)
        chan = MappingFilter(chan)
        sigma = chan['fitResults_sigma']
        # error_sigma = chan['fitError_sigma']
        theta = chan['fitResults_theta']
        error_theta = chan['fitError_theta']
        lobesep = chan['fitResults_lobesep']
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

        # make sure we don't have a pi jump in the middle of our spline!
        unwrapped_theta_cal = np.unwrap(np.asarray(cal['theta'])[z_valid_mask], np.pi/2, period=np.pi)
        theta_spline = LSQUnivariateSpline(z_valid, unwrapped_theta_cal, knots, ext='const')
        theta_cal = theta_spline(z_v)
        dtheta_dz = theta_spline.derivative()  # Note -> constant region of theta cal spline will have dtheta_dz=0

        lobesep_spline = LSQUnivariateSpline(z_valid, np.array(cal['lobesep'])[z_valid_mask], knots, ext='const')
        sig_spline = LSQUnivariateSpline(z_valid, np.array(cal['sigma'])[z_valid_mask], knots, ext='const')


        z_out = np.empty_like(theta)
        # error_z_out = np.empty_like(theta)  # we'll do this vectorized after the loop
        # lobesep_residual = np.empty_like(theta)  # we'll do this vectorized after the loop
        # sigma_residual = np.empty_like(theta)  # we'll do this vectorized after the loop

        # loop over each localization and find it's Z position
        for ind in range(len(theta)):
            # use sin in the residual calc to handle wrapping
            theta_residual = np.sin(theta[ind] - theta_cal)  # [rad., under small angle approximation]
            theta_normed = theta_residual ** 2

            # min_loc = np.argmin(np.stack([
            #     # sigma_normed, 
            #     theta_normed, 
            #     lobesep_normed
            # ], axis=0).sum(axis=0))
            min_loc = np.argmin(theta_normed)

            z_out[ind] = z_v[min_loc]
        
        error_z_out = error_theta * (1 / dtheta_dz(z_out))  # err_z = |dz/dtheta| * err_theta, [nm] = [rad] * [nm/rad]
        error_z_out[np.isinf(error_z_out)] = np.finfo(np.float32).max  # replace any inf from flat theta with maxfloat
        # look-up the lobesep and sigma residuals, accounting for possible sign differences
        # lobesep_residual = np.abs(lobesep) - np.abs(lobesep_spline(z_out))
        # sigma_residual = np.abs(sigma) - np.abs(sig_spline(z_out))
        lobesep_residual = lobesep - lobesep_spline(z_out)
        sigma_residual = sigma - sig_spline(z_out)  # FIXME - should we be taking absolute value on sigma?


        amplitudes = np.stack([chan['fitResults_A0'], chan['fitResults_A1']], axis=1)
        amplitudes = np.sort(amplitudes, axis=1)
        chan.addColumn('dh_amp_ratio', amplitudes[:,0] / amplitudes[:,1])

        chan.addColumn('dh_z', z_out)
        chan.addColumn('dh_z_error', error_z_out)
        chan.addColumn('dh_lobesep_residual', lobesep_residual)
        chan.addColumn('dh_sigma_residual', sigma_residual)
        chan.addColumn('dh_xy_detection_residual', np.sqrt((chan['x'] - chan['startParams_x0'])**2 + (chan['y'] - chan['startParams_y0'])**2))

        if c_ind < 1:
            dh_loc = chan
        else:
            dh_loc = ConcatenateFilter(dh_loc, chan)
        
        if plot: # store z lookedup splines for plotting
            theta_cals.append(theta_cal)

            lobesep_cal = lobesep_spline(z_v)
            lobesep_cals.append(lobesep_cal)

            sig_cal = sig_spline(z_v)
            sig_cals.append(sig_cal)

    dh_loc.setMapping('z', 'dh_z + z')
    
    if not plot:
        return dh_loc
    else:

        return dh_loc, Plot(lambda: plot_dh_z_lookup([{
            'theta': theta_cals[c_ind],
            'sigma': sig_cals[c_ind],
            'lobesep': lobesep_cals[c_ind],
            'z_v': z_v
        } for c_ind in range(len(calibration))], dh_loc))

def plot_dh_z_lookup(calibration_splines, dh_loc):
    from matplotlib import pyplot as plt
    from PYME.IO.tabular import ColourFilter
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    for c_ind, cal in enumerate(calibration_splines):
        # grab localizations corresponding to this channel
        chan = ColourFilter(dh_loc, currentColour=c_ind)

        sigma = chan['fitResults_sigma']
        error_sigma = chan['fitError_sigma']
        theta = chan['fitResults_theta']
        error_theta = chan['fitError_theta']
        lobesep = chan['fitResults_lobesep']
        error_lobesep = chan['fitError_lobesep']
        z_out = chan['dh_z']
        z_v = cal['z_v']

        ax[0].plot(z_v, cal['theta'], ':', label='Splined Cal.')
        ax[0].errorbar(z_out, theta, error_theta, linestyle='')

        ax[1].plot(z_v, cal['sigma'], ':', label='Splined Cal.')
        ax[1].errorbar(z_out, sigma, error_sigma, linestyle='')

        ax[2].plot(z_v, cal['lobesep'], ':', label='Splined Cal.')
        ax[2].errorbar(z_out, lobesep, error_lobesep, linestyle='')
    
    ax[0].set_ylabel('Theta [rad]')
    ax[0].legend()

    ax[1].set_ylabel('Sigma [nm]')
    ax[1].legend()

    ax[2].set_ylabel('Lobe Separation [nm]')
    ax[2].legend()
    
    plt.xlabel('Z [nm]')

    return fig


def computeStrengthMap(sigma, image):

    """Compute second derivative of Gaussian steerable filter strength map

    Parameters
    ----------
    image : array
       single image to be filtered
    sigma: float
        sigma of the Gaussian used to generate 7 basis filters

    Returns
    -------
    strengthMap
        filtered image, not noramlized 
    """
    
    from double_helix.DoubleGaussFit import g2a, g2b, g2c, h2a, h2b, h2c, h2d

    roiHalfSize = 10 # [pixels]

    xx = np.mgrid[(-roiHalfSize):(roiHalfSize + 1)]
    yy = np.mgrid[(-roiHalfSize):(roiHalfSize + 1)]


    X, Y = xx[:, None], yy[None, :]

    f1 = g2a(Y, X, sigma)
    f2 = g2b(Y, X, sigma)
    f3 = g2c(Y, X, sigma)

    g2a_xy = ndimage.convolve(image, f1)
    g2b_xy = ndimage.convolve(image, f2)
    g2c_xy = ndimage.convolve(image, f3)
    
    fha = h2a(Y, X, sigma)
    fhb = h2b(Y, X, sigma)
    fhc = h2c(Y, X, sigma)
    fhd = h2d(Y, X, sigma)

    h2a_xy = ndimage.convolve(image, fha)
    h2b_xy = ndimage.convolve(image, fhb)
    h2c_xy = ndimage.convolve(image, fhc)
    h2d_xy = ndimage.convolve(image, fhd)

    c_2= 0.5 * (g2a_xy**2 - g2c_xy**2) \
                + 0.46875*(h2a_xy**2 - h2d_xy**2) \
                + 0.28125*(h2b_xy**2 - h2c_xy**2) \
                + 0.1875 * (h2a_xy*h2c_xy - h2b_xy * h2d_xy)
    c_3 = - g2a_xy*g2b_xy - g2b_xy * g2c_xy \
                - 0.9375 * (h2c_xy * h2d_xy + h2a_xy * h2b_xy) \
                - 1.6875 * h2b_xy * h2c_xy - 0.1875 * h2a_xy * h2d_xy

    
    strengthMap = np.sqrt(c_2 ** 2 + c_3 ** 2)
    
    return strengthMap

def StrengthVsSigma(sigmas, image):
        
    strengths = np.asarray(np.zeros(sigmas.size), dtype=float)
    
    for i in list(range(0,sigmas.size)):
    
        strengthMap = computeStrengthMap(sigmas[i], image)
        strengths[i] = np.max(strengthMap)
     
    
    optSigma = int(sigmas[np.unravel_index(np.argmax(strengths), strengths.shape)[0]])
    
    return optSigma, strengths

def OptimalSigmabyFrame(PSFStack, sigmas):
    
    """Compute second derivative of Gaussian steerable filter strength map

    Parameters
    ----------
    PSFStack : PYME.IO.image.ImageStack
       image of single, extracted PSF, with even z spacing between slices. If
       one is working with imagestacks at uneven Z or with multiple frames per
       step, one should first average by z step, and then resample the PSF stack
       single image to be filtered
    sigmas : array
        array of filter sigmas to test to find optimal filter sigma

    Returns
    -------
    optSigmas
        array of optimal sigma value for each step in z stack
    
    strengths
        strength of optimized filter response at each step of z stack

    maxX, maxY
        arrays containing the x and y coordinates respectively that correspond to the position 
        of the maximized filter response for each step in z stack

    """

    optSigmas = np.asarray(np.zeros(PSFStack.shape[2]), dtype=float)
    maxX = np.asarray(np.zeros(PSFStack.shape[2]), dtype=float)
    maxY = np.asarray(np.zeros(PSFStack.shape[2]), dtype=float)
    
    for j in list(range(0,PSFStack.shape[2])):
    
        strengths = np.asarray(np.zeros(sigmas.size), dtype=float)

        X = np.asarray(np.zeros(sigmas.size), dtype=float)
        Y = np.asarray(np.zeros(sigmas.size), dtype=float)
        
        for i in list(range(0,sigmas.size)):

            
            strengthMap = computeStrengthMap(sigmas[i], PSFStack[:,:,j])
            strengths[i] = np.max(strengthMap)
            X[i] = np.unravel_index(np.argmax(strengthMap), strengthMap.shape)[1]
            Y[i] = np.unravel_index(np.argmax(strengthMap), strengthMap.shape)[0]

        optSigmas[j] = sigmas[np.unravel_index(np.argmax(strengths), strengths.shape)[0]]
        maxX[j] = X[np.unravel_index(np.argmax(strengths), strengths.shape)[0]]
        maxY[j] = Y[np.unravel_index(np.argmax(strengths), strengths.shape)[0]]
        
    
    return optSigmas, strengths, maxX, maxY


def StrengthVsZ(PSFStack, sigma, l, s):
    
    """Compute second derivative of Gaussian steerable filter strength map

    Parameters
    ----------
    PSFStack : PYME.IO.image.ImageStack
       image of single, extracted PSF, with even z spacing between slices. If
       one is working with imagestacks at uneven Z or with multiple frames per
       step, one should first average by z step, and then resample the PSF stack
       single image to be filtered
    sigma : array
        sigma of Gaussian used to generate 7 basis filters
    l : float
        lobe separation (px) of DHPSF to use for computing normalization factor
    s : float
        sigma (px) of DHPSF lobe to use for computing normalization factor

    Returns
    -------
    
    strengths
        normalized strength of filter response at each step of z stack
    
    dhAmplitudes
        max value of DHPSF in each z slice

    maxX, maxY
        arrays containing the x and y coordinates respectively that correspond to the position 
        of the maximized filter response for each step in z stack

    """
    num_frames = PSFStack.shape[2]

    A=1
    S = lambda x: pow(2.718281828459045,3)*pow(3.141592653589793,3)*pow(pow(A,4)*pow(s,8)*pow(x,16)*pow((pow(s,2)+pow(x,2)),-6)*pow((-1*pow(2.718281828459045,-0.25*pow((pow(s,2)+pow(x,2)),-1))*pow((math.erf(0.5*(-1+l)*pow(2,-0.5)*pow((pow(s,2)+pow(x,2)),-0.5))+-1*math.erf(0.5*(1+l)*pow(2,-0.5)*pow((pow(s,2)+pow(x,2)),-0.5))),2)+pow(2.718281828459045,-0.25*pow((1+l),2)*pow((pow(s,2)+pow(x,2)),-1))*pow(math.erf(0.5*pow(2,-0.5)*pow((pow(s,2)+pow(x,2)),-0.5)),2)*pow((1+l+-1*(-1+l)*pow(2.718281828459045,0.5*l*pow((pow(s,2)+pow(x,2)),-1))),2)),2),0.5)
    normFactor = S(sigma)
    
    strengths = np.asarray(np.zeros(PSFStack.shape[2]), dtype=float)
    dhAmplitudes = np.asarray(np.zeros(PSFStack.shape[2]), dtype=float)
    maxX = np.asarray(np.zeros(PSFStack.shape[2]), dtype=float)
    maxY = np.asarray(np.zeros(PSFStack.shape[2]), dtype=float)
    
    for i in list(range(0,PSFStack.shape[2])):
    
        strengthMap = computeStrengthMap(sigma, PSFStack[:,:,i])
        strengths[i] = np.max(np.sqrt(strengthMap/normFactor))
        dhAmplitudes[i] = np.max(PSFStack[:,:,i])
        maxX[i] = np.unravel_index(np.argmax(strengthMap), strengthMap.shape)[1]
        maxY[i] = np.unravel_index(np.argmax(strengthMap), strengthMap.shape)[0]

    return strengths, dhAmplitudes, maxX, maxY