#!/usr/bin/python

##################
# LatGaussFitFR.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import numpy as np
from scipy import ndimage
from PYME.localization.FitFactories.fitCommon import fmtSlicesUsed, pack_results
from PYME.localization.FitFactories import FFBase 
from PYME.Analysis._fithelpers import FitModelWeighted, FitModelWeightedJac

##################
# Model functions
def f_dh(p, X, Y, bgd=None):
    """Model function for double helix (two gaussians) parameterized on x,y center, lobe
    separation, and lobe angle.

    Parameters
    ----------
    p : list
        parameter vector:
            amplitude0: amplitude of one of the Gaussians [ADU]
            amplitude1: amplitude of the second Gaussian [ADU]
            x: x center position [nm]
            y: y center position [nm]
            theta: angle created by two lobes [radians]
            lobe_separation: distance between centers of the two gaussians [nm]
            sigma: Gaussian sigma, shared parameter of both Gaussians [nm]
            constant_background_offset: constant background term added to model function [ADU]
    X : np.ndarray
        x grid positions (1D, will broadcast to 2D)
    Y : np.ndarray
        y grid positions (1D, will broadcast to 2D)
    bgd : ndarray, optional
        per-pixel background estimate from e.g. sliding-window methods, by default None.
        bgd gets added onto the model function, rather than subtracted from the incoming
        data to preserve the noise model of the fit.

    Returns
    -------
    ndarray
        model function evaluated with parameters p
    """
    # amplitude0, amplitude1, x, y, theta, lobe_separation, sigma, constant_background_offset
    A0, A1, x, y, theta, lobe_sep, sig, bg = p
    X = X[:,None]
    Y = Y[None,:]
    arg0x = -((X - x + 0.5 * lobe_sep * np.cos(theta)) ** 2) / (2 * sig ** 2)
    arg0y = -((Y - y + 0.5 * lobe_sep * np.sin(theta)) ** 2) / (2 * sig ** 2)
    arg1x = -((X - x - 0.5 * lobe_sep * np.cos(theta)) ** 2) / (2 * sig ** 2)
    arg1y = -((Y - y - 0.5 * lobe_sep * np.sin(theta)) ** 2) / (2 * sig ** 2)
    f = bg + A0 * np.exp(arg0x + arg0y) + A1 * np.exp(arg1x + arg1y)
    
    if bgd is not None:
        f += bgd   
    return f

def f_dh_jac(p, X, Y, bgd=None):
    """first row of jacobian (partial first derivatives of f_dh for each var in p)

    Parameters
    ----------
    p : list
        parameter vector:
            amplitude0: amplitude of one of the Gaussians [ADU]
            amplitude1: amplitude of the second Gaussian [ADU]
            x: x center position [nm]
            y: y center position [nm]
            theta: angle created by two lobes [radians]
            lobe_separation: distance between centers of the two gaussians [nm]
            sigma: Gaussian sigma, shared parameter of both Gaussians [nm]
            constant_background_offset: constant background term added to model function [ADU]
    X : np.ndarray
        x grid positions (1D, will broadcast to 2D)
    Y : np.ndarray
        y grid positions (1D, will broadcast to 2D)
    bgd : np.ndarray
        per-pixel background estimate, if determined separately. 
        Not used here at the moment, which effectively says we
        assume it is 'flat enough' not to matter here. But passing
        it to be compatible with PYME _fithelpers weightedjac function
    
    Returns
    -------
    ndarray
        the Jacobian of f_dh across the rows, unraveled and stack for compatibility with 
        PYME.Analysis._fithelpers.weightedJacF
    """
    # amplitude0, amplitude1, x, y, theta, lobe_separation, sigma, constant_background_offset
    A0, A1, x, y, theta, lobe_sep, sig, bg = p
    X = X[:,None]
    Y = Y[None,:]
    denom = (2 * sig ** 2)
    sqrt_arg0x_num = X - x + 0.5 * lobe_sep * np.cos(theta)
    sqrt_arg0y_num = Y - y + 0.5 * lobe_sep * np.sin(theta)
    sqrt_arg1x_num = X - x - 0.5 * lobe_sep * np.cos(theta)
    sqrt_arg1y_num = Y - y - 0.5 * lobe_sep * np.sin(theta)
    
    ja0 = np.exp(-(sqrt_arg0x_num ** 2)/denom - (sqrt_arg0y_num ** 2)/denom)
    ja1 = np.exp(-(sqrt_arg1x_num ** 2)/denom - (sqrt_arg1y_num ** 2)/denom)

    sigpre = (1/(sig ** 2))
    jx = sigpre * (A0 * sqrt_arg0x_num * ja0 + A1 * sqrt_arg1x_num * ja1)
    
    jy = sigpre * (A0 * sqrt_arg0y_num * ja0 + A1 * sqrt_arg1y_num * ja1)
    
    jtheta = A0 * ja0 * sigpre * 0.5 * lobe_sep * (np.sin(theta) * sqrt_arg0x_num - np.cos(theta) * sqrt_arg0y_num)
    jtheta += A1 * ja1 * sigpre * 0.5 * lobe_sep * (np.cos(theta) * sqrt_arg1y_num - np.sin(theta) * sqrt_arg1x_num)

    jlobe_sep = -A0 * ja0 * sigpre * 0.5 * (np.sin(theta) * sqrt_arg0y_num + np.cos(theta) * sqrt_arg0x_num)
    jlobe_sep += A1 * ja1 * sigpre * 0.5 * (np.cos(theta) * sqrt_arg1x_num + np.sin(theta) * sqrt_arg1y_num)

    jsig = A0 * ja0 * (1/(sig ** 3)) * (sqrt_arg0x_num ** 2 + sqrt_arg0y_num ** 2)
    jsig += A1 * ja1 * (1/(sig ** 3)) * (sqrt_arg1x_num ** 2 + sqrt_arg1y_num ** 2)

    jbg = np.ones_like(ja0)
    # return [ja0.ravel(), ja1.ravel(), jx.ravel(), jy.ravel(), jtheta.ravel(), jlobe_sep.ravel(), jsig.ravel(), jbg.ravel()]
    return np.stack([ja0.ravel(), ja1.ravel(), jx.ravel(), jy.ravel(), jtheta.ravel(), jlobe_sep.ravel(), jsig.ravel(), jbg.ravel()], axis=1)

# store the derivative function as an attribute so we can toggle below
# (not super important)
f_dh.D = f_dh_jac

#####################

#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A0', '<f4'), ('A1', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
                              ('theta', '<f4'),
                              ('lobesep', '<f4'),
                              ('sigma', '<f4'), 
                              ('background', '<f4')]),
              ('fitError', [('A0', '<f4'), ('A1', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
                              ('theta', '<f4'),
                              ('lobesep', '<f4'),
                              ('sigma', '<f4'), 
                              ('background', '<f4')]),
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4'),
              ('startParams', [('A0', '<f4'), ('A1', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
                              ('theta', '<f4'),
                              ('lobesep', '<f4'),
                              ('sigma', '<f4'), 
                              ('background', '<f4')]),
              ]

def FitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, background=0, length = 0):
    slicesUsed = fmtSlicesUsed(slicesUsed)
    #print slicesUsed

    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')
    
    
    res =  np.array([(metadata.tIndex, fitResults.astype('f'), fitErr.astype('f'), length, resultCode, slicesUsed, background)], dtype=fresultdtype) 
    #print res
    return res

_dh_detector = None


# see pg 39, http://robots.stanford.edu/cs223b04/SteerableFiltersfreeman91design.pdf
def g2a(x, y):
    return 0.9213 * (2 * x**2 - 1) * np.exp(- (x**2 + y**2))
def g2b(x, y):
    return 1.843*x*y*np.exp(- (x**2 + y**2))
def g2c(x, y):
    return 0.9213 * (2 * y**2 - 1) * np.exp(- (x**2 + y**2))
# separatable basis set and interpolation functions for fit to hilbert transform of second derivative of gaussian
def h2a(x, y):
    return 0.9780  * (-2.254 * x + x ** 3) * np.exp( - (x**2 + y**2))
def h2b(x, y):
    return 0.9780 * (-0.7515 + x ** 2) * y * np.exp( - (x**2 + y**2))
def h2c(x, y):
    return 0.9780 * (-0.7515 + y ** 2) * x * np.exp( - (x**2 + y**2))
def h2d(x, y):
    return 0.9780  * (-2.254 * y + y ** 3) * np.exp( - (x**2 + y**2))


class Detector(object):
    def __init__(self, roi_half_size, mag=1.0):
        """
        Parameters:
        -----------
        roi_half_size: int
            ROI size used in max filtering and 
            filter kernel generation will be
            2 * roi_half_size + 1
        mag: float
            scale to resize convolution kernel pixels relative to input image
            pixels.
        
        Notes:
        ------
        Create this object, call filter_frame method followed by extract_candidates
        on each frame you want to detect double helix PSFs.
        """
        self.mag = mag
        self.roi_half_size = roi_half_size
        xx = np.mgrid[(-roi_half_size):(roi_half_size + 1)]
        yy = np.mgrid[(-roi_half_size):(roi_half_size + 1)]
        X, Y = mag * xx[:, None], mag * yy[None, :]
        self.X = X
        self.Y = Y

        self.roi_half_size = roi_half_size
        self.roi_size = 2 * roi_half_size + 1

        self.g2a = g2a(X, Y)
        self.g2b = g2b(X, Y)
        self.g2c = g2c(X, Y)

        self.h2a = h2a(X, Y)
        self.h2b = h2b(X, Y)
        self.h2c = h2c(X, Y)
        self.h2d = h2d(X, Y)

    def filter_frame(self, image):
        """
        Parameters:
        -----------
        image: ndarray
            2D, single frame image
        """
        # print('here - image size: %s, g2a size: %s' % (image.shape, self.g2a.shape))
        g2a_xy = ndimage.convolve(image, self.g2a)
        g2b_xy = ndimage.convolve(image, self.g2b)
        g2c_xy = ndimage.convolve(image, self.g2c)

        h2a_xy = ndimage.convolve(image, self.h2a)
        h2b_xy = ndimage.convolve(image, self.h2b)
        h2c_xy = ndimage.convolve(image, self.h2c)
        h2d_xy = ndimage.convolve(image, self.h2d)

        c_2= 0.5 * (g2a_xy**2 - g2c_xy**2) \
                    + 0.46875*(h2a_xy**2 - h2d_xy**2) \
                    + 0.28125*(h2b_xy**2 - h2c_xy**2) \
                    + 0.1875 * (h2a_xy*h2c_xy - h2b_xy * h2d_xy)
        c_3 = - g2a_xy*g2b_xy - g2b_xy * g2c_xy \
                    - 0.9375 * (h2c_xy * h2d_xy + h2a_xy * h2b_xy) \
                    - 1.6875 * h2b_xy * h2c_xy - 0.1875 * h2a_xy * h2d_xy

        strength_image = np.sqrt(c_2 ** 2 + c_3 ** 2)
        angle_image = np.arctan2(c_3, c_2) / 2

        return strength_image, angle_image
    
    def extract_candidates(self, strength_image, angle_image, threshold):
        """
        Returns:
        --------
        row: ndarray
            row positions [pix] of candidate emitters
        col: ndarray
            col positions [pix] of candidate emitters
        angle: ndarray
            orientation [radians] of candidate emitters. 0 points along row in
            increasing col.
        """
        max_filtered_strength = ndimage.maximum_filter(strength_image, 
                                                        (self.roi_size,
                                                         self.roi_size))
        
        candidate_image = np.logical_and(max_filtered_strength == strength_image, strength_image >= threshold)
        print(np.where(candidate_image))
        row, col = np.where(candidate_image)
        angle = np.empty_like(row, dtype=float)  # radians
        for ind in range(len(row)):
            angle[ind] = angle_image[row[ind], col[ind]]
        return row, col, angle


# def lobe_estimate_from_center_pixel(x_pix, y_pix, orientation, lobe_sep_px):
#     dx = np.cos(orientation) * lobe_sep_px * 0.5
#     dy = np.sin(orientation) * lobe_sep_px * 0.5
#     x1 = x_pix - dx
#     y1 = y_pix - dy
#     x2 = x_pix + dx
#     y2 = y_pix + dy
#     return x1, y1, x2, y2

		

class DumbellFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_dh, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma)

        if 'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else: 
            self.solver = FitModelWeighted
    
    def refresh_detector(self):

        global _dh_detector # One instance for each process, re-used for subsequent fits.

        # guess_psf_sigma_pix = self.metadata.getOrDefault('Analysis.GuessPSFSigmaPix',
        #                                                  600 / 2.8 / (self.metadata.voxelsize_nm.x))
        # make one at the end, otherwise if we're OK return early
        need_fresh = False
        if not _dh_detector:
            need_fresh = True  # we don't have one yet
        else:
            need_fresh = _dh_detector.mag != self.metadata.getEntry('Analysis.DetectionFilterMag') or _dh_detector.roi_half_size != self.metadata.getEntry('Analysis.ROISize')
        
        if need_fresh:
            _dh_detector = Detector(self.metadata.getEntry('Analysis.ROISize'), mag=self.metadata.getEntry('Analysis.DetectionFilterMag'))
        return
    
    def FindAndFit(self, threshold, cameraMaps, **kwargs):
        """

        Parameters
        ----------
        threshold: float
            detection threshold, as a multiple of per-pixel noise standard deviation
        cameraMaps: cameraInfoManager object (see remFitBuf.py)

        Returns
        -------
        results

        """
        # make sure we've built correct filters
        self.refresh_detector()

        # at this point, data is in ADU, with offset subtracted and flatfield applied
        
        # Find candidate molecule positions on background-subtracted frame
        if self.background is not None:
            bgd = (self.data.astype('f') - self.background).squeeze()
        else:
            bgd = self.data.astype('f').squeeze()
        # print(bgd.shape)
        # print(self.noiseSigma.shape)
        
        # Note PYME flips row/col y/x, so feed the detector a Transposed frame to get it 'right'
        strength_image, angle_image = _dh_detector.filter_frame(bgd.T)

        row, col, orientation = _dh_detector.extract_candidates(strength_image, angle_image, threshold * self.noiseSigma.squeeze())

        # lobe_sep_pix = self.metadata.getEntry('Analysis.LobeSepGuess') / self.metadata.voxelsize_nm.x
        # x0, y0, x1, y1 = lobe_estimate_from_center_pixel(col, row, orientation, lobe_sep_pix)
        # convert positions from pixels to nm
        x_nm = self.metadata.voxelsize_nm.x * (col + self.roi_offset[0])
        y_nm = self.metadata.voxelsize_nm.y * (row + self.roi_offset[1])
        # x0_nm = self.metadata.voxelsize_nm.x * (x0 + self.roi_offset[0])
        # x1_nm = self.metadata.voxelsize_nm.x * (x1 + self.roi_offset[0])
        # y0_nm = self.metadata.voxelsize_nm.y * (y0 + self.roi_offset[1])
        # y1_nm = self.metadata.voxelsize_nm.y * (y1 + self.roi_offset[1])
        

        #PYME ROISize is a half size
        roi_half_size = self.metadata.getEntry('Analysis.ROISize')  # int(2*self.metadata.getEntry('Analysis.ROISize') + 1)
        lobe_sep = self.metadata.getEntry('Analysis.LobeSepGuess')  # [nm]
        sigma_guess = self.metadata.getEntry('Analysis.SigmaGuess')  # [nm]
        ########## Actually do the fits #############
        n_cand = len(row)
        results = np.empty(n_cand, FitResultsDType)

        for ind in range(n_cand):
            x_pix = col[ind]
            y_pix = row[ind]
            X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x_pix, y_pix, None, roi_half_size)

            dataMean = data - background

            
            amp = (data - data.min()).max() #amplitude

            # vs = self.metadata.voxelsize_nm
            # x0 =  vs.x*x
            # y0 =  vs.y*y
            
            bgm = np.mean(background)
            guess = (amp, amp, x_nm[ind], y_nm[ind], orientation[ind], lobe_sep, sigma_guess, dataMean.min())
            # guess = (amp, x0_nm[ind], y0_nm[ind], amp, x1_nm[ind], y1_nm[ind], 160, dataMean.min())
            
            #do the fit
            (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, guess, data, sigma, X, Y, background)

            #try to estimate errors based on the covariance matrix
            fit_errors=None
            try:       
                fit_errors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
            except Exception:
                pass
            
            # length = np.sqrt((res[1] - res[4])**2 + (res[2] - res[5])**2)
            # x_com = 0.5 * (res[1] + res[4])
            # y_com = 0.5 * (res[2] + res[5])
            # theta = np.arctan2(res[4] - res[1], res[5] - res[2])
            
            if False:
                #display for debugging purposes
                import matplotlib.pyplot as plt
                plt.figure(figsize=(20, 5))
                plt.subplot(151)
                plt.title('Background')
                plt.imshow(background)
                plt.colorbar()
                plt.subplot(152)
                plt.title('Background Sub')
                plt.imshow(dataMean)
                plt.colorbar()
                plt.subplot(153)
                plt.title('Init. Guess')
                plt.imshow(f_dh(guess, X, Y))
                plt.colorbar()
                plt.subplot(154)
                plt.title('Fitted Results')
                plt.imshow(f_dh(res, X, Y))
                plt.colorbar()
                plt.subplot(155)
                plt.title('Residuals')
                plt.imshow(dataMean-f_dh(res, X, Y))
                plt.colorbar()

            #package results
            results[ind] = pack_results(FitResultsDType, self.metadata.tIndex, res, fit_errors, startParams=guess, slicesUsed=(xslice, yslice, zslice), 
                                resultCode=resCode, subtractedBackground=bgm)  # , length=length, x=x_com, y=y_com, theta=theta)
            # results[ind] = FitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm, length)
        
        return results

    def FromPoint(self, x, y, z=None, roiHalfSize=7, axialHalfSize=0):
        roiHalfSize = self.metadata.getOrDefault('Analysis.ROISize', roiHalfSize)
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background
        
        # make sure we've built correct filters
        self.refresh_detector()
        # at this point, data is in ADU, with offset subtracted and flatfield applied
        
        # Find candidate molecule positions on background-subtracted frame
        if self.background is not None:
            bgd = (self.data.astype('f') - self.background).squeeze()
        else:
            bgd = self.data.astype('f').squeeze()
        # print(bgd.shape)
        # print(self.noiseSigma.shape)
        
        # Note PYME flips row/col y/x, so feed the detector a Transposed frame to get it 'right'
        strength_image, angle_image = _dh_detector.filter_frame(bgd.T)
        # assume the max of the image is what we're supposed to fit
        # row, col = np.argmax(strength_image, axis=0), np.argmax(strength_image, axis=1)
        row, col = np.where(strength_image == strength_image.max())
        row, col = row[0], col[0]
        orientation = angle_image[row, col]
        # row, col, orientation = _dh_detector.extract_candidates(strength_image, angle_image, threshold * self.noiseSigma.squeeze())

        # lobe_sep_pix = self.metadata.getEntry('Analysis.LobeSepGuess') / self.metadata.voxelsize_nm.x
        # x0, y0, x1, y1 = lobe_estimate_from_center_pixel(col, row, orientation, lobe_sep_pix)
        # convert positions from pixels to nm
        x_nm = self.metadata.voxelsize_nm.x * (col + self.roi_offset[0])
        y_nm = self.metadata.voxelsize_nm.y * (row + self.roi_offset[1])
        # x0_nm = self.metadata.voxelsize_nm.x * (x0 + self.roi_offset[0])
        # x1_nm = self.metadata.voxelsize_nm.x * (x1 + self.roi_offset[0])
        # y0_nm = self.metadata.voxelsize_nm.y * (y0 + self.roi_offset[1])
        # y1_nm = self.metadata.voxelsize_nm.y * (y1 + self.roi_offset[1])
        

        #PYME ROISize is a half size
        # roi_half_size = self.metadata.getEntry('Analysis.ROISize')  # int(2*self.metadata.getEntry('Analysis.ROISize') + 1)
        lobe_sep = self.metadata.getEntry('Analysis.LobeSepGuess')  # [nm]
        sigma_guess = self.metadata.getEntry('Analysis.SigmaGuess')  # [nm]
        ########## Actually do the fits #############
        # n_cand = len(row)
        # results = np.empty(n_cand, FitResultsDType)

        # for ind in range(n_cand):
        # x_pix = col
        # y_pix = row
            # X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x_pix, y_pix, None, roi_half_size)

            # dataMean = data - background

            
        amp = (data - data.min()).max() #amplitude

        # vs = self.metadata.voxelsize_nm
        # x0 =  vs.x*x
        # y0 =  vs.y*y
        
        bgm = np.mean(background)
        guess = (amp, amp, x_nm, y_nm, orientation, lobe_sep, sigma_guess, dataMean.min())
        # guess = (amp, x0_nm[ind], y0_nm[ind], amp, x1_nm[ind], y1_nm[ind], 160, dataMean.min())
        
        #do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, guess, data, sigma, X, Y, background)

        #try to estimate errors based on the covariance matrix
        fit_errors=None
        try:       
            fit_errors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception:
            pass
        
        # length = np.sqrt((res[1] - res[4])**2 + (res[2] - res[5])**2)
        # x_com = 0.5 * (res[1] + res[4])
        # y_com = 0.5 * (res[2] + res[5])
        # theta = np.arctan2(res[4] - res[1], res[5] - res[2])
        
        if False:
            #display for debugging purposes
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 5))
            plt.subplot(151)
            plt.title('Background')
            plt.imshow(background)
            plt.colorbar()
            plt.subplot(152)
            plt.title('Background Sub')
            plt.imshow(dataMean)
            plt.colorbar()
            plt.subplot(153)
            plt.title('Init. Guess')
            plt.imshow(f_dh(guess, X, Y))
            plt.colorbar()
            plt.subplot(154)
            plt.title('Fitted Results')
            plt.imshow(f_dh(res, X, Y))
            plt.colorbar()
            plt.subplot(155)
            plt.title('Residuals')
            plt.imshow(dataMean-f_dh(res, X, Y))
            plt.colorbar()

        #package results
        return pack_results(FitResultsDType, self.metadata.tIndex, res, fit_errors, startParams=guess, slicesUsed=(xslice, yslice, zslice), 
                                resultCode=resCode, subtractedBackground=bgm)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        """Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation"""
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_dh(params, X, Y), X[0], Y[0], 0)


#so that fit tasks know which class to use
FitFactory = DumbellFitFactory
FitResult = FitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

MULTIFIT=True # weird way to say it, but flag that this module does its own detection

import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.IntParam('Analysis.ROISize', u'ROI half size', 10),
    # mde.BoolParam('Analysis.GPUPCTBackground', 'Calculate percentile background on GPU', True),
    mde.FloatParam('Analysis.DetectionFilterMag', 'Detection Filter Scaling Magnification:', 0.15,
                 'Currently the steerable filter is defined with a sigma=1 pix filter, so manually scale it to match your double helix PSF'),
    mde.FloatParam('Analysis.LobeSepGuess', 'Double Helix Lobe Separation Guess [nm]:', 900,
                   'What lobe separation should the fit expect, and therefore begin with?'),
    mde.FloatParam('Analysis.SigmaGuess', 'Double Helix Lobe Sigma Guess [nm]:', 180,
                   'Estimate for Gaussian sigma parameter to initialize the fitting')
]

DESCRIPTION = 'Fit a Double Helix two-Gaussian function'
LONG_DESCRIPTION = 'Fit a "dumbell" consisting of 2 Gaussians'
USE_FOR = '3D using a Double-Helix point-spread function'