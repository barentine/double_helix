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
def f_dumbell(p, X, Y, bgd=None):
    """
    bgd: ndarray, optional
        per-pixel background if previously estimated, to be
        added onto the model so it doesn't have to be subtracted
        and mess up the noise model.
    """
    A, x0, y0, B, x1, y1, s, bg = p
    X = X[:,None]
    Y = Y[None,:]
    r = A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + B*np.exp(-((X-x1)**2 + (Y - y1)**2)/(2*s**2)) + bg 
    if bgd is not None:
        r += bgd
    #print r.shape    
    return r

def f_dh(p, X, Y, bgd=None):
    # amplitude0, amplitude1, x, y, theta, lobe_separation, sigma, constant_background_offset
    A0, A1, x, y, theta, lobe_sep, sig, bg = p
    # FIXME

def f_dh_jac(p, X, Y):
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
    """
    # amplitude0, amplitude1, x, y, theta, lobe_separation, sigma, constant_background_offset
    A0, A1, x, y, theta, lobe_sep, sig, bg = p
    # FIXME

f_dh.D = f_dh_jac

#####################

#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
                              ('B', '<f4'),
                              ('x1', '<f4'),('y1', '<f4'),
                              ('sigma', '<f4'), 
                              ('background', '<f4')]),
              ('fitError', [('A', '<f4'),
                            ('x0', '<f4'),
                            ('y0', '<f4'),
                            ('B', '<f4'),
                            ('x1', '<f4'),('y1', '<f4'),
                            ('sigma', '<f4'), 
                            ('background', '<f4')]),
            #   ('length', '<f4'),
            #   ('x', '<f4'),
            #   ('y', '<f4'),
            #   ('theta', '<f4'),
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4')
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


def lobe_estimate_from_center_pixel(x_pix, y_pix, orientation, lobe_sep_px):
    dx = np.cos(orientation) * lobe_sep_px * 0.5
    dy = np.sin(orientation) * lobe_sep_px * 0.5
    x1 = x_pix - dx
    y1 = y_pix - dy
    x2 = x_pix + dx
    y2 = y_pix + dy
    return x1, y1, x2, y2

		

class DumbellFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_dumbell, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma)

        if False:#'D' in dir(fitfcn): #function has jacobian
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
        bgd = (self.data.astype('f') - self.background).squeeze()
        print(bgd.shape)
        print(self.noiseSigma.shape)
        
        # Note PYME flips row/col y/x, so feed the detector a Transposed frame to get it 'right'
        strength_image, angle_image = _dh_detector.filter_frame(bgd.T)

        row, col, orientation = _dh_detector.extract_candidates(strength_image, angle_image, threshold * self.noiseSigma.squeeze())

        lobe_sep_pix = self.metadata.getEntry('Analysis.LobeSepGuess') / self.metadata.voxelsize_nm.x
        x0, y0, x1, y1 = lobe_estimate_from_center_pixel(col, row, orientation, lobe_sep_pix)
        # convert these positions from pixels to nm
        x0_nm = self.metadata.voxelsize_nm.x * (x0 + self.roi_offset[0])
        x1_nm = self.metadata.voxelsize_nm.x * (x1 + self.roi_offset[0])
        y0_nm = self.metadata.voxelsize_nm.y * (y0 + self.roi_offset[1])
        y1_nm = self.metadata.voxelsize_nm.y * (y1 + self.roi_offset[1])

        #PYME ROISize is a half size
        roi_half_size = self.metadata.getEntry('Analysis.ROISize')  # int(2*self.metadata.getEntry('Analysis.ROISize') + 1)

        ########## Actually do the fits #############
        n_cand = len(row)
        results = np.empty(n_cand, FitResultsDType)

        for ind in range(n_cand):
            x = col[ind]
            y = row[ind]
            X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, None, roi_half_size)

            dataMean = data - background

            
            amp = (data - data.min()).max() #amplitude

            # vs = self.metadata.voxelsize_nm
            # x0 =  vs.x*x
            # y0 =  vs.y*y
            
            bgm = np.mean(background)

            guess = (amp, x0_nm[ind], y0_nm[ind], amp, x1_nm[ind], y1_nm[ind], 160, dataMean.min())
            
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
                plt.imshow(f_dumbell(guess, X, Y))
                plt.colorbar()
                plt.subplot(154)
                plt.title('Fitted Results')
                plt.imshow(f_dumbell(res, X, Y))
                plt.colorbar()
                plt.subplot(155)
                plt.title('Residuals')
                plt.imshow(dataMean-f_dumbell(res, X, Y))
                plt.colorbar()

            #package results
            results[ind] = pack_results(FitResultsDType, self.metadata.tIndex, res, fit_errors, startParams=guess, slicesUsed=(xslice, yslice, zslice), 
                                resultCode=resCode, subtractedBackground=bgm)  # , length=length, x=x_com, y=y_com, theta=theta)
            # results[ind] = FitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm, length)
        
        return results

    def FromPoint(self, x, y, z=None, roiHalfSize=7, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background
        
        #print data.shape

        #estimate some start parameters...
        A = (data - data.min()).max() #amplitude

        vs = self.metadata.voxelsize_nm
        x0 =  vs.x*x
        y0 =  vs.y*y
        
        bgm = np.mean(background)

        startParameters = [A, x0 + 70*np.random.randn(), y0+ 70*np.random.randn(), A, x0+ 70*np.random.randn(), y0+ 70*np.random.randn(), 250/2.35, dataMean.min()]	

        
        #do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y)

        #try to estimate errors based on the covariance matrix
        fitErrors=None
        try:       
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception:
            pass
        
        # length = np.sqrt((res[1] - res[4])**2 + (res[2] - res[5])**2)
        
        if False:
            #display for debugging purposes
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.subplot(141)
            plt.imshow(dataMean)
            plt.colorbar()
            plt.subplot(142)
            plt.imshow(f_dumbell(startParameters, X, Y))
            plt.colorbar()
            plt.subplot(143)
            plt.imshow(f_dumbell(res, X, Y))
            plt.colorbar()
            plt.subplot(144)
            plt.imshow(dataMean-f_dumbell(res, X, Y))
            plt.colorbar()

        #package results
        return FitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm)#, length)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        """Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation"""
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_dumbell(params, X, Y), X[0], Y[0], 0)


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
                   'What lobe separation should the fit expect, and therefore begin with?')
]

DESCRIPTION = 'Fit a "dumbell" consisting of 2 Gaussians'
LONG_DESCRIPTION = 'Fit a "dumbell" consisting of 2 Gaussians'
USE_FOR = '2D single-colour'