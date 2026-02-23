
from PYME.IO import MetaDataHandler
import numpy as np
from double_helix import DoubleGaussFit
from scipy import ndimage

md = MetaDataHandler.SimpleMDHandler()
md['Analysis.ROISize'] = 10
md['voxelsize.units'] = 'um'
md['voxelsize.x'] = 0.12
md['voxelsize.y'] = 0.12
md['voxelsize.z'] = 0.1
md['Analysis.SigmaGuess'] = 200  # Double Helix Lobe Sigma Guess [nm]
md['Analysis.LobeSepGuess'] = 1050  # Double Helix Lobe Separation Guess [nm]
md['Analysis.DetectionFilterSigma'] = 5  # Detection Filter Sigma (in px)
# A0, A1, x, y, theta, lobe_sep, sig, bg = p
md['Test.DefaultParams'] = [500, 500, 0, 0, np.radians(17.243), md['Analysis.LobeSepGuess'], md['Analysis.SigmaGuess'], 20]
md['Test.ParamJitter'] = [50, 50, 120, 120, 0.5*np.pi, 20, 10, 10]
md['Test.ROISize'] = md['Analysis.ROISize']

test_im, _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(md['Test.DefaultParams'], md, 
                                          x=md['voxelsize.x']*md['Test.ROISize'], 
                                          y=md['voxelsize.y']*md['Test.ROISize'], roiHalfSize=md['Test.ROISize'])

xx = np.arange(- md['Test.ROISize'], md['Test.ROISize'] + 1)
yy = np.arange(- md['Test.ROISize'], md['Test.ROISize'] + 1)
X = xx[:, None]
Y = yy[None, :]

def test_g2a_separability():


        # xx = np.mgrid[(-roi_half_size):(roi_half_size + 1)]
        # yy = np.mgrid[(-roi_half_size):(roi_half_size + 1)]
        # X, Y = xx[:, None], yy[None, :]

    g2a_2d_kernel = DoubleGaussFit.g2a(X, Y, sig=md['Analysis.DetectionFilterSigma'])
    g2a_xy_2d = ndimage.convolve(test_im, g2a_2d_kernel)

    g2a_x_kernel = DoubleGaussFit.g2a_x(xx, sig=md['Analysis.DetectionFilterSigma'])
    g2a_y_kernel = DoubleGaussFit.g2a_y(yy, sig=md['Analysis.DetectionFilterSigma'])
    g2a_x = ndimage.convolve1d(test_im, g2a_x_kernel, axis=0)
    g2a_xy_sep = ndimage.convolve1d(g2a_x, g2a_y_kernel, axis=1)

    np.testing.assert_almost_equal(g2a_xy_2d, g2a_xy_sep)

def test_g2b_separability():

    g2b_2d_kernel = DoubleGaussFit.g2b(X, Y, sig=md['Analysis.DetectionFilterSigma'])
    g2b_xy_2d = ndimage.convolve(test_im, g2b_2d_kernel)

    g2b_x_kernel = DoubleGaussFit.g2b_x(xx, sig=md['Analysis.DetectionFilterSigma'])
    g2b_y_kernel = DoubleGaussFit.g2b_y(yy, sig=md['Analysis.DetectionFilterSigma'])
    g2b_x = ndimage.convolve1d(test_im, g2b_x_kernel, axis=0)
    g2b_xy_sep = ndimage.convolve1d(g2b_x, g2b_y_kernel, axis=1)

    np.testing.assert_almost_equal(g2b_xy_2d, g2b_xy_sep)

def test_g2c_separability():
    g2c_2d_kernel = DoubleGaussFit.g2c(X, Y, sig=md['Analysis.DetectionFilterSigma'])
    g2c_xy_2d = ndimage.convolve(test_im, g2c_2d_kernel)

    g2c_x_kernel = DoubleGaussFit.g2c_x(xx, sig=md['Analysis.DetectionFilterSigma'])
    g2c_y_kernel = DoubleGaussFit.g2c_y(yy, sig=md['Analysis.DetectionFilterSigma'])
    g2c_x = ndimage.convolve1d(test_im, g2c_x_kernel, axis=0)
    g2c_xy_sep = ndimage.convolve1d(g2c_x, g2c_y_kernel, axis=1)

    np.testing.assert_almost_equal(g2c_xy_2d, g2c_xy_sep)

def test_h2a_separability():
    h2a_2d_kernel = DoubleGaussFit.h2a(X, Y, sig=md['Analysis.DetectionFilterSigma'])
    h2a_xy_2d = ndimage.convolve(test_im, h2a_2d_kernel)

    h2a_x_kernel = DoubleGaussFit.h2a_x(xx, sig=md['Analysis.DetectionFilterSigma'])
    h2a_y_kernel = DoubleGaussFit.h2a_y(yy, sig=md['Analysis.DetectionFilterSigma'])
    h2a_x = ndimage.convolve1d(test_im, h2a_x_kernel, axis=0)
    h2a_xy_sep = ndimage.convolve1d(h2a_x, h2a_y_kernel, axis=1)

    np.testing.assert_almost_equal(h2a_xy_2d, h2a_xy_sep)

def test_h2b_separability():
    h2b_2d_kernel = DoubleGaussFit.h2b(X, Y, sig=md['Analysis.DetectionFilterSigma'])
    h2b_xy_2d = ndimage.convolve(test_im, h2b_2d_kernel)

    h2b_x_kernel = DoubleGaussFit.h2b_x(xx, sig=md['Analysis.DetectionFilterSigma'])
    h2b_y_kernel = DoubleGaussFit.h2b_y(yy, sig=md['Analysis.DetectionFilterSigma'])
    h2b_x = ndimage.convolve1d(test_im, h2b_x_kernel, axis=0)
    h2b_xy_sep = ndimage.convolve1d(h2b_x, h2b_y_kernel, axis=1)

    np.testing.assert_almost_equal(h2b_xy_2d, h2b_xy_sep)

def test_h2c_separability():
    h2c_2d_kernel = DoubleGaussFit.h2c(X, Y, sig=md['Analysis.DetectionFilterSigma'])
    h2c_xy_2d = ndimage.convolve(test_im, h2c_2d_kernel)

    h2c_x_kernel = DoubleGaussFit.h2c_x(xx, sig=md['Analysis.DetectionFilterSigma'])
    h2c_y_kernel = DoubleGaussFit.h2c_y(yy, sig=md['Analysis.DetectionFilterSigma'])
    h2c_x = ndimage.convolve1d(test_im, h2c_x_kernel, axis=0)
    h2c_xy_sep = ndimage.convolve1d(h2c_x, h2c_y_kernel, axis=1)

    np.testing.assert_almost_equal(h2c_xy_2d, h2c_xy_sep)

def test_h2d_separability():
    h2d_2d_kernel = DoubleGaussFit.h2d(X, Y, sig=md['Analysis.DetectionFilterSigma'])
    h2d_xy_2d = ndimage.convolve(test_im, h2d_2d_kernel)

    h2d_x_kernel = DoubleGaussFit.h2d_x(xx, sig=md['Analysis.DetectionFilterSigma'])
    h2d_y_kernel = DoubleGaussFit.h2d_y(yy, sig=md['Analysis.DetectionFilterSigma'])
    h2d_x = ndimage.convolve1d(test_im, h2d_x_kernel, axis=0)
    h2d_xy_sep = ndimage.convolve1d(h2d_x, h2d_y_kernel, axis=1)

    np.testing.assert_almost_equal(h2d_xy_2d, h2d_xy_sep)
