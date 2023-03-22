
import numpy as np
from scipy import ndimage

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
    def __init__(self, image_size, roi_half_size, mag=1.0):
        """
        Parameters:
        -----------
        image_size: list (int)
            row, col size of images to detect on
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

        row, col = np.where(candidate_image)
        angle = np.empty_like(row, dtype=float)  # radians
        for ind in range(len(row)):
            angle[ind] = angle_image[row[ind], col[ind]]
        return row, col, angle
