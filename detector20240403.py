
import numpy as np
import math
from scipy import ndimage

# see pg 39, http://robots.stanford.edu/cs223b04/SteerableFiltersfreeman91design.pdf
def g2a(x, y, sig):
    return (1 / (2 * sig**2)) * np.exp((3/2)) * (x**2 - sig**2) * np.exp(- (x**2 + y**2) / (2 * sig**2))
def g2b(x, y, sig):
    return (1 / (2 * sig**2)) * np.exp((3/2)) * x*y*np.exp(- (x**2 + y**2) / (2 * sig**2))
def g2c(x, y, sig):
    return (1 / (2 * sig**2)) * np.exp((3/2)) * (y**2 - sig**2) * np.exp(- (x**2 + y**2) / (2 * sig**2))
# separatable basis set and interpolation functions for fit to hilbert transform of second derivative of gaussian
def h2a(x, y, sig):
    return (np.exp((3/2)) / (24 * np.pi * sig**3)) * np.exp(- (x**2 + y**2) / (2 * sig**2)) * (4 * (2 * np.pi)**(1/2) * x**3 - 3 * sig * (8 * (2 * np.pi)**(1/2) * sig - (np.log(1/sig**2) + np.log(sig**2))) * x)
def h2b(x, y, sig):
    return (np.exp((3/2)) / (24 * np.pi * sig**3)) * np.exp(- (x**2 + y**2) / (2 * sig**2)) * (4 * (2 * np.pi)**(1/2) * x**2 - sig * (8 * (2 * np.pi)**(1/2) * sig - (np.log(1/sig**2) + np.log(sig**2)))) * y
def h2c(x, y, sig):
    return (np.exp((3/2)) / (24 * np.pi * sig**3)) * np.exp(- (x**2 + y**2) / (2 * sig**2)) * (4 * (2 * np.pi)**(1/2) * y**2 - sig * (8 * (2 * np.pi)**(1/2) * sig - (np.log(1/sig**2) + np.log(sig**2)))) * x
def h2d(x, y, sig):
    return (np.exp((3/2)) / (24 * np.pi * sig**3)) * np.exp(- (x**2 + y**2) / (2 * sig**2)) * (4 * (2 * np.pi)**(1/2) * y**3 - 3 * sig * (8 * (2 * np.pi)**(1/2) * sig - (np.log(1/sig**2) + np.log(sig**2))) * y)


class Detector(object):
    def __init__(self, roi_half_size=10, l_initial=900, lobe_sigma_initial=180, filter_sigma=8.0, pxSize=117.4):
        """
        Parameters:
        -----------
        roi_half_size: int
            ROI size used in max filtering and 
            filter kernel generation will be
            2 * roi_half_size + 1
        l_initial: float
            initial guess for lobe separation
        lobe_sigma_initial: float
            initial guess for lobe sigma    
        filter_sigma: float
            sigma for g2 and h2 filters in pixels
        pxSize: float
            pixel size in nm
        
        Notes:
        ------
        Create this object, call filter_frame method followed by extract_candidates
        on each frame you want to detect double helix PSFs.
        """
        self.roi_half_size = roi_half_size
        self.filter_sigma = filter_sigma
        self.pxSize = pxSize
        xx = np.mgrid[(-roi_half_size):(roi_half_size + 1)]
        yy = np.mgrid[(-roi_half_size):(roi_half_size + 1)]
        X, Y = xx[:, None], yy[None, :]
        self.X = X
        self.Y = Y

        self.roi_half_size = roi_half_size
        self.roi_size = 2 * roi_half_size + 1

        self.g2a = g2a(X, Y, filter_sigma)
        self.g2b = g2b(X, Y, filter_sigma)
        self.g2c = g2c(X, Y, filter_sigma)

        self.h2a = h2a(X, Y, filter_sigma)
        self.h2b = h2b(X, Y, filter_sigma)
        self.h2c = h2c(X, Y, filter_sigma)
        self.h2d = h2d(X, Y, filter_sigma)

        # compute normalization factor for strength image
        # analytically computed value for max filter response for given dhpsf and filter sigma
        A=1
        l = l_initial/(self.pxSize)
        s = lobe_sigma_initial/(self.pxSize)
        S = lambda x: pow(2.718281828459045,3)*pow(3.141592653589793,3)*pow(pow(A,4)*pow(s,8)*pow(x,16)*pow((pow(s,2)+pow(x,2)),-6)*pow((-1*pow(2.718281828459045,-0.25*pow((pow(s,2)+pow(x,2)),-1))*pow((math.erf(0.5*(-1+l)*pow(2,-0.5)*pow((pow(s,2)+pow(x,2)),-0.5))+-1*math.erf(0.5*(1+l)*pow(2,-0.5)*pow((pow(s,2)+pow(x,2)),-0.5))),2)+pow(2.718281828459045,-0.25*pow((1+l),2)*pow((pow(s,2)+pow(x,2)),-1))*pow(math.erf(0.5*pow(2,-0.5)*pow((pow(s,2)+pow(x,2)),-0.5)),2)*pow((1+l+-1*(-1+l)*pow(2.718281828459045,0.5*l*pow((pow(s,2)+pow(x,2)),-1))),2)),2),0.5)
        self.normFactor = S(filter_sigma)

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

        strength_image = np.sqrt((np.sqrt(c_2 ** 2 + c_3 ** 2)/self.normFactor))
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
