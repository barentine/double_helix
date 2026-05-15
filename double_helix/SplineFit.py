#!/usr/bin/python

##################
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
from PYME.localization.FitFactories.fitCommon import fmtSlicesUsed, pack_results
from PYME.localization.FitFactories import FFBase 
from PYME.Analysis._fithelpers import FitModelWeighted_, FitModelWeightedJac

from double_helix.DoubleGaussFit import Detector
from PYME.localization.FitFactories import InterpFitR
from PYME.localization.FitFactories.Interpolators import CSInterpolator
import logging
logger = logging.getLogger(__name__)

_dh_detector = None
_start_pos_estimator = None


class ZEstimator(object):
    def __init__(self):
        self.splines = {}
    
    def calibrate(self, interpolator, metadata):
        from scipy.interpolate import UnivariateSpline
        # generate grid for evaluation
        roi_size = metadata.getEntry('Analysis.ROISize')
        X, Y, Z, safe_region = interpolator.getCoords(metadata, 
                                                      slice(-roi_size, roi_size), 
                                                      slice(-roi_size, roi_size), 
                                                      slice(0, 2))
        # get Z range from interpolator model extent, excluding boundary slices
        # that cInterp cannot evaluate (cubic spline requires ~3 index margin)
        z_vals = interpolator.IntZVals[3:-3]
        n_z = len(z_vals)
        z = np.empty(n_z)
        angle = np.empty(n_z)
        # need to instantiate a detector to pull the orientation estimate. Assume the brightest strength is at the center and extract
        # that angle
        calibration_detector = Detector(metadata.getEntry('Analysis.ROISize'),
                                        l_initial=metadata.getEntry('Analysis.LobeSepGuess'),
                                        lobe_sigma_initial=metadata.getEntry('Analysis.SigmaGuess'),
                                        filter_sigma=metadata.getEntry('Analysis.DetectionFilterSigma'),
                                        px_size_nm=metadata.voxelsize_nm.x)
        
        # # evaluate PSF on grid
        for z_ind, z_val in enumerate(z_vals):
            z[z_ind] = z_val
            slice_data = interpolator.interp(X, Y, Z + z_val).squeeze()
            strength_image, angle_image = calibration_detector.filter_frame(slice_data.T)  # keep the transpose to match with FindAndFit
            row, col = np.where(strength_image == strength_image.max())
            angle[z_ind] = angle_image[row[0], col[0]]
        
        # spline z as a function of angle 
        spline = UnivariateSpline(angle, z)
        self.splines['z'] = spline
    
    def estimate_z_from_orientation(self, orientation):
        if not 'z' in self.splines.keys():
            raise ValueError('Z estimator not calibrated')
        
        return self.splines['z'](orientation)


class PSFFitFactory(FFBase.FFBase):
    def __init__(self, data, metadata, fitfcn=InterpFitR.f_Interp3d, background=None, noiseSigma=None, **kwargs):
        super(PSFFitFactory, self).__init__(data, metadata, fitfcn, background, noiseSigma, **kwargs)
        
        if 'D' in dir(fitfcn):
            self.solver = FitModelWeightedJac
        else:
            self.solver = FitModelWeighted_
        
        # interp_module = metadata.getOrDefault('Analysis.InterpModule', 'CSInterpolator')
        self.interpolator = CSInterpolator.interpolator

        global _start_pos_estimator
        if _start_pos_estimator is None:
            _start_pos_estimator = ZEstimator()
        self.startPosEstimator = _start_pos_estimator

        if self.interpolator.setModelFromMetadata(metadata):
            logger.info('model changed')
            self.startPosEstimator.splines.clear()

        if not 'z' in self.startPosEstimator.splines.keys():
            # we need to get a "z" vs theta spline 
            self.startPosEstimator.calibrate(self.interpolator, metadata)
            
    
    def estimate_z_from_orientation(self, orientation):
        return self.startPosEstimator.estimate_z_from_orientation(orientation)

    def refresh_detector(self):

        global _dh_detector # One instance for each process, re-used for subsequent fits.

        # make one at the end, otherwise if we're OK return early
        need_fresh = False
        if not _dh_detector:
            need_fresh = True  # we don't have one yet
        else:
            need_fresh = _dh_detector.roi_half_size != self.metadata.getEntry('Analysis.ROISize') or _dh_detector.filter_sigma != self.metadata.getEntry('Analysis.DetectionFilterSigma')
        
        if need_fresh:
            _dh_detector = Detector(self.metadata.getEntry('Analysis.ROISize'),
                                    l_initial=self.metadata.getEntry('Analysis.LobeSepGuess'),
                                    lobe_sigma_initial=self.metadata.getEntry('Analysis.SigmaGuess'),
                                    filter_sigma=self.metadata.getEntry('Analysis.DetectionFilterSigma'),
                                    px_size_nm=self.metadata.voxelsize_nm.x)
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

        # negative values in bg subtraction lead to detection artefacts so make any negative pixel have value of 0
        bgd[bgd<0] = 0
        
        # Note PYME flips row/col y/x, so feed the detector a Transposed frame to get it 'right'
        strength_image, angle_image = _dh_detector.filter_frame(bgd.T)

        # _dh_detector.normFactor * (threshold * self.noiseSigma.squeeze())**2 allows for thresholding on
        #   values similar to SNR
        row, col, orientation = _dh_detector.extract_candidates(strength_image, angle_image, _dh_detector.normFactor * (threshold * self.noiseSigma.T.squeeze())**2)

        # lobe_sep_pix = self.metadata.getEntry('Analysis.LobeSepGuess') / self.metadata.voxelsize_nm.x
        # x0, y0, x1, y1 = lobe_estimate_from_center_pixel(col, row, orientation, lobe_sep_pix)
        # convert positions from pixels to nm
        x_nm = self.metadata.voxelsize_nm.x * (col + self.roi_offset[0])
        y_nm = self.metadata.voxelsize_nm.y * (row + self.roi_offset[1])
        # x0_nm = self.metadata.voxelsize_nm.x * (x0 + self.roi_offset[0])
        # x1_nm = self.metadata.voxelsize_nm.x * (x1 + self.roi_offset[0])
        # y0_nm = self.metadata.voxelsize_nm.y * (y0 + self.roi_offset[1])
        # y1_nm = self.metadata.voxelsize_nm.y * (y1 + self.roi_offset[1])
        
        z_nm = self.estimate_z_from_orientation(orientation)

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
            X, Y, Z, safeRegion = self.interpolator.getCoords(self.metadata, xslice, yslice, zslice)
            dataMean = data - background
            
            amp = (data - data.min()).max() #amplitude

            # vs = self.metadata.voxelsize_nm
            # x0 =  vs.x*x
            # y0 =  vs.y*y
            
            bgm = np.mean(background)
            guess = (amp, x_nm[ind], y_nm[ind], z_nm[ind], dataMean.min())
            
            #do the fit
            # (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, guess, data, sigma, X, Y, background)
            (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, guess, dataMean, sigma, self.interpolator, X, Y, Z, safeRegion)


            #try to estimate errors based on the covariance matrix
            fit_errors=None
            try:       
                fit_errors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
            except Exception:
                pass
            
            #normalised Chi-squared
            nchi2 = (infodict['fvec']**2).sum()/(dataMean.size - res.size)
            
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
                                resultCode=resCode, subtractedBackground=bgm, nchi2=nchi2)  # , length=length, x=x_com, y=y_com, theta=theta)
        
        return results

FitFactory = PSFFitFactory
FitResult = InterpFitR.FitResult
FitResultsDType = InterpFitR.FitResultsDType

MULTIFIT=True # flag that this module does its own detection

import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.IntParam('Analysis.ROISize', u'ROI half size', 10),
    mde.FilenameParam('PSFFile', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf|TIFF files|*.tif|H5 files|*.h5'),
    mde.FloatParam('Analysis.DetectionFilterSigma', 'Detection Filter Sigma (in px):', 8.0,
                 'scales filters spatially'),
    mde.FloatParam('Analysis.LobeSepGuess', 'Double Helix Lobe Separation Guess [nm]:', 900,
                   'What lobe separation should the detector use for intensity normalization?'),
    mde.FloatParam('Analysis.SigmaGuess', 'Double Helix Lobe Sigma Guess [nm]:', 180,
                   'Estimate for Gaussian sigma parameter to use for intensity normalization during detection')
]

DESCRIPTION = '3D, double helix fitting using an interpolated PSF.'
LONG_DESCRIPTION = DESCRIPTION
USE_FOR = '3D using a Double-Helix point-spread function'