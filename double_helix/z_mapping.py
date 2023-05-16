
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

def calibrate_double_helix_psf(image, fit_module, roi_half_size=11):
    #TODO - move all non-GUI logic for this out of this file?
    from PYME.recipes.measurement import FitPoints
    # from PYME.Analysis.PSFEst import extractImages


    # from PYME.IO.FileUtils import nameUtils
    # import matplotlib.pyplot as plt
    # import matplotlib.cm
    # import json
    # query user for type of calibration
    # NB - GPU fit is not enabled here because it exits on number of iterations, which is not necessarily convergence for very bright beads!
    # ftypes = ['BeadConvolvedAstigGaussFit', 'AstigGaussFitFR']  # , 'AstigGaussGPUFitFR']
    # fitType_dlg = wx.SingleChoiceDialog(self.dsviewer, 'Fit-type selection', 'Fit-type selection', ftypes)
    # fitType_dlg.ShowModal()
    # fitMod = ftypes[fitType_dlg.GetSelection()]

    # if (fitMod == 'BeadConvolvedAstigGaussFit') and ('Bead.Diameter' not in self.image.mdh.keys()):
    #     beadDiam_dlg = wx.NumberEntryDialog(None, 'Bead diameter in nm', 'diameter [nm]', 'diameter [nm]', 100, 1, 9e9)
    #     beadDiam_dlg.ShowModal()
    #     beadDiam = float(beadDiam_dlg.GetValue())
    #     # store this in metadata
    #     self.image.mdh['Analysis.Bead.Diameter'] = beadDiam



    # At the moment, this z spacing to be even along the PSF.
    # one should first average by z, and then resample PSF stack, then finally
    # run this calibration. 
    vs_x_nm = image.voxelsize_nm.x
    vs_y_nm = image.voxelsize_nm.y
    z_step_nm = image.voxelsize_nm.z
    n_steps = image.data_xyztc.shape[2]
    obj_positions = {}

    obj_positions['x'] = vs_x_nm * 0.5 * image.data_xyztc.shape[0] * np.ones(n_steps)
    obj_positions['y'] = vs_y_nm * 0.5 * image.data_xyztc.shape[1] * np.ones(n_steps)
    obj_positions['t'] = np.arange(image.data.shape[2])
    z = np.arange(image.data_xyztc.shape[2]) * image.mdh['voxelsize.z'] * 1.e3
    obj_positions['z'] = z - z.mean()

    results = []

    for chan_ind in range(image.data_xyztc.shape[3]):
        # get z centers
        # dx, dy, dz = extractImages.getIntCenter(image.data_xyztc[:, :, :, chan_ind])

        res = FitPoints().apply_simple(inputImage=image, 
                                       inputPositions=obj_positions,
                                       roiHalfSize=roi_half_size,
                                       fitModule=fit_module, channel=chan_ind)

        # dsigma = abs(res['fitResults_sigmax']) - abs(res['fitResults_sigmay'])
        # valid = ((res['fitError_sigmax'] > 0) * (res['fitError_sigmax'] < 50)* (res['fitError_sigmay'] < 50)*(res['fitResults_A'] > 0) > 0)
        results.append([
            {
                'theta': res['fitResults_theta'].tolist(),
                'lobesep': res['fitResults_lobesep'].tolist(),
                'sigma': res['fitResults_lobesep'].tolist(),
                'z': obj_positions['z'].tolist(),
            }
        ])
        # results.append({'sigmax': abs(res['fitResults_sigmax'][valid]).tolist(),'error_sigmax': abs(res['fitError_sigmax'][valid]).tolist(),
                        # 'sigmay': abs(res['fitResults_sigmay'][valid]).tolist(), 'error_sigmay': abs(res['fitError_sigmay'][valid]).tolist(),
                        # 'dsigma': dsigma[valid].tolist(), 'z': obj_positions['z'][valid].tolist(), 'zCenter': obj_positions['z'][int(dz)]})

    #generate new tab to show results
    # use_web_view = False
    # if not '_astig_view' in dir(self):
    #     try:
    #         self._astig_view= wx.html2.WebView.New(self.dsviewer)
    #         self.dsviewer.AddPage(self._astig_view, True, 'Astigmatic calibration')

    #     except NotImplementedError:
    #         use_web_view = False

    # find reasonable z range for each channel, inject 'zRange' into the results. FIXME - injection is bad
    # results = astigTools.find_and_add_zRange(results)

    #do plotting
    # plt.ioff()
    # f = plt.figure(figsize=(10, 4))

    # colors = iter(matplotlib.cm.Dark2(np.linspace(0, 1, 2*self.image.data.shape[3])))
    # plt.subplot(121)
    # for i, res in enumerate(results):
    #     nextColor1 = next(colors)
    #     nextColor2 = next(colors)
    #     lbz = np.absolute(res['z'] - res['zRange'][0]).argmin()
    #     ubz = np.absolute(res['z'] - res['zRange'][1]).argmin()
    #     plt.plot(res['z'], res['sigmax'], ':', c=nextColor1)  # , label='x - %d' % i)
    #     plt.plot(res['z'], res['sigmay'], ':', c=nextColor2)  # , label='y - %d' % i)
    #     plt.plot(res['z'][lbz:ubz], res['sigmax'][lbz:ubz], label='x - %d' % i, c=nextColor1)
    #     plt.plot(res['z'][lbz:ubz], res['sigmay'][lbz:ubz], label='y - %d' % i, c=nextColor2)

    # #plt.ylim(-200, 400)
    # plt.grid()
    # plt.xlabel('z position [nm]')
    # plt.ylabel('Sigma [nm]')
    # plt.legend()

    # plt.subplot(122)
    # colors = iter(matplotlib.cm.Dark2(np.linspace(0, 1, self.image.data.shape[3])))
    # for i, res in enumerate(results):
    #     nextColor = next(colors)
    #     lbz = np.absolute(res['z'] - res['zRange'][0]).argmin()
    #     ubz = np.absolute(res['z'] - res['zRange'][1]).argmin()
    #     plt.plot(res['z'], res['dsigma'], ':', lw=2, c=nextColor)  # , label='Chan %d' % i)
    #     plt.plot(res['z'][lbz:ubz], res['dsigma'][lbz:ubz], lw=2, label='Chan %d' % i, c=nextColor)
    # plt.grid()
    # plt.xlabel('z position [nm]')
    # plt.ylabel('Sigma x - Sigma y [nm]')
    # plt.legend()

    # plt.tight_layout()

    # plt.ion()
    # #dat = {'z' : objPositions['z'][valid].tolist(), 'sigmax' : res['fitResults_sigmax'][valid].tolist(),
    # #                   'sigmay' : res['fitResults_sigmay'][valid].tolist(), 'dsigma' : dsigma[valid].tolist()}


    # if use_web_view:
    #     fig = mpld3.fig_to_html(f)
    #     data = json.dumps(results)

    #     template = env.get_template('astigCal.html')
    #     html = template.render(astigplot=fig, data=data)
    #     #print html
    #     self._astig_view.SetPage(html, '')
    # else:
    #     plt.show()

    # fdialog = wx.FileDialog(None, 'Save Astigmatism Calibration as ...',
    #     wildcard='Astigmatism Map (*.am)|*.am', style=wx.FD_SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath())  #, defaultFile=defFile)
    # succ = fdialog.ShowModal()
    # if (succ == wx.ID_OK):
    #     fpath = fdialog.GetPath()

    #     fid = open(fpath, 'w', encoding='utf8')
    #     json.dump(results, fid, indent=4, sort_keys=True)
    #     fid.close()
    #     if use_web_view:  # save the html too
    #         import os
    #         fpath = os.path.splitext(fpath)[0] + '.html'
    #         with open(fpath, 'wb') as fid:
    #             fid.write(html.encode('utf-8'))

    return results

def lookup_astig_z(fres, astig_calibrations, rough_knot_spacing=75., plot=False):
    """
    Generates a look-up table of sorts for z based on sigma x/y fit results and calibration information. If a molecule
    appears on multiple planes, sigma values from both planes will be used in the look up.

    Parameters
    ----------
    fres : dict-like
        Contains fit results (localizations) to be mapped in z
    astig_calibrations : list
        Each element is a dictionary corresponding to a multiview channel, which contains the x and y PSF widths at
        various z-positions
    rough_knot_spacing : Float
        Smoothing is applied to the sigmax/y look-up curves by fitting a cubic spline with knots spaced roughly at
        intervals of rough_knot_spacing (in nanometers). There is potentially rounding within the step-size of the
        astigmatism calibration to make knot placing more convenient.
    plot : bool
        Flag to toggle plotting

    Returns
    -------
    z : ndarray
        astigmatic Z-position of each localization in fres
    zerr : ndarray
        discrepancies between sigma values and the PSF calibration curves

    """
    # fres = pipeline.selectedDataSource.resultsSource.fitResults
    # numMolecules = len(fres['x']) # there is no guarantee that fitResults_x0 will be present - change to x
    numChans = len(astig_calibrations)

    # find overall min and max z values
    z_min = 0
    z_max = 0
    for astig_cal in astig_calibrations: #more idiomatic way of looping through list - also avoids one list access / lookup
        r_min, r_max = astig_cal['zRange']
        z_min = min(z_min, r_min)
        z_max = max(z_max, r_max)

    # generate z vector for interpolation
    zVal = np.arange(z_min, z_max)

    # generate look up table of sorts
    sigCalX = []
    sigCalY = []
    for i, astig_cal in enumerate(astig_calibrations):
        zdat = np.array(astig_cal['z'])

        # grab indices of range we trust
        z_range = astig_cal['zRange']
        z_valid_mask = (zdat > z_range[0])*(zdat < z_range[1])
        z_valid = zdat[z_valid_mask]

        # generate splines with knots spaced roughly as rough_knot_spacing [nm]
        z_steps = np.unique(z_valid)
        dz_med = np.median(np.diff(z_steps))
        smoothing_factor = int(rough_knot_spacing / (dz_med))
        knots = z_steps[1:-1:smoothing_factor]

        sigCalX.append(LSQUnivariateSpline(z_valid,np.array(astig_cal['sigmax'])[z_valid_mask], knots, ext='const')(zVal))
        sigCalY.append(LSQUnivariateSpline(z_valid,np.array(astig_cal['sigmay'])[z_valid_mask], knots, ext='const')(zVal))

    sigCalX = np.array(sigCalX)
    sigCalY = np.array(sigCalY)

    # #allocate arrays for the estimated z positions and their errors
    # z = np.zeros(numMolecules)
    # zerr = 1e4 * np.ones(numMolecules)
    #
    # failures = 0
    chans = np.arange(numChans)

    #extract our sigmas and their errors
    #doing this here means we only do the string operations and look-ups once, rather than once per molecule
    s_xs = np.abs(np.array([fres['sigmax%i' % ci] for ci in chans]))
    s_ys = np.abs(np.array([fres['sigmay%i' % ci] for ci in chans]))
    esxs = [fres['error_sigmax%i' % ci] for ci in chans]
    esys = [fres['error_sigmay%i' % ci] for ci in chans]
    wXs = np.array([1. / (esx_i*esx_i) for esx_i in esxs])
    wYs = np.array([1. / (esy_i*esy_i) for esy_i in esys])

    if plot:
        from matplotlib import pyplot as plt

        plt.figure()
        plt.subplot(211)
        for astig_cal, interp_sigx, col in zip(astig_calibrations, sigCalX, ['r', 'g', 'b', 'c']):
            plt.plot(astig_cal['z'], astig_cal['sigmax'], ':', c=col)
            plt.plot(zVal, interp_sigx, c=col)

        plt.subplot(212)
        for astig_cal, interp_sigy, col in zip(astig_calibrations, sigCalY, ['r', 'g', 'b', 'c']):
            plt.plot(astig_cal['z'], astig_cal['sigmay'], ':', c=col)
            plt.plot(zVal, interp_sigy, c=col)

    # _lenz_chunked = np.floor(len(zVal)) - 1
    # sigCalX_chunked = np.ascontiguousarray(sigCalX[:,::100])
    # sigCalY_chunked = np.ascontiguousarray(sigCalY[:,::100])
    #
    # for i in range(numMolecules):
    #     #TODO - can we avoid this loop?
    #     wX = wXs[:, i]
    #     wY = wYs[:, i]
    #     sx = sxs[:, i]
    #     sy = sys[:, i]
    #
    #     wSum = (wX + wY).sum()
    #
    #     #estimate the position in two steps - coarse then fine
    #
    #     #coarse step:
    #     errX = (wX[:,None] * (sx[:, None] - sigCalX_chunked)**2).sum(0)
    #     errY = (wY[:, None] * (sy[:, None] - sigCalY_chunked)**2).sum(0)
    #
    #     err = (errX + errY) / wSum
    #     loc_coarse = min(max(np.argmin(err), 1), _lenz_chunked)
    #
    #     fine_s =  100*(loc_coarse - 1)
    #     fine_end = 100*(loc_coarse + 1)
    #
    #     #print loc_coarse, fine_s, fine_end, sigCalX.shape
    #
    #     #fine step
    #     errX = (wX[:, None] * (sx[:, None] - sigCalX[:,fine_s:fine_end]) ** 2).sum(0)
    #     errY = (wY[:, None] * (sy[:, None] - sigCalY[:,fine_s:fine_end]) ** 2).sum(0)
    #
    #     err = (errX + errY) / wSum
    #     minLoc = np.argmin(err)
    #
    #     z[i] = -zVal[fine_s + minLoc]
    #     zerr[i] = np.sqrt(err[minLoc])

    zi, ze = astiglookup.astig_lookup(sigCalX.T.astype('f'), sigCalY.T.astype('f'), s_xs.T.astype('f'),
                                      s_ys.T.astype('f'), wXs.T.astype('f'), wYs.T.astype('f'))

    print('used c lookup')

    z = -zVal[zi]
    zerr = np.sqrt(ze)


    #print('%i localizations did not have sigmas in acceptable range/planes (out of %i)' % (failures, numMolecules))

    return z, zerr