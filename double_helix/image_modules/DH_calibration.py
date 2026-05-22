
import numpy
from PYME.DSView.modules._base import Plugin
import wx
import logging
logger = logging.getLogger(__name__)


def _xy_centroid(im_2d):
    """
    Thresholded intensity-weighted centroid of a 2D array.

    Returns (dx, dy) offsets from the array centre, matching the threshold
    logic used by PYME.Analysis.PSFEst.extractImages.getIntCenter.
    """
    import numpy as np
    im = np.asarray(im_2d, dtype='f').squeeze()
    im2 = im - im.min()
    im2 = im2 - 0.5 * im2.max()
    im2 = np.maximum(im2, 0)
    total = im2.sum()
    if total < 1e-9:
        return 0.0, 0.0
    nx, ny = im2.shape
    X = np.arange(nx, dtype='f') - (nx - 1) / 2.0
    Y = np.arange(ny, dtype='f') - (ny - 1) / 2.0
    dx = float((im2 * X[:, None]).sum() / total)
    dy = float((im2 * Y[None, :]).sum() / total)
    return dx, dy


def _extract_dh_psf(data_3d, locs, center_slice, roi_half_xy, blur):
    """
    Extract a DH PSF by averaging XY-centered sub-stacks from multiple bead locations.

    For each tagged bead a full-z ROI is extracted.  The XY centroid is estimated
    on z= `center_slice`, and the beads sub-stacks are laterally aligned with
    sub-pixel precision by applying a phase ramp to each sub-stack in Fourier space
    before averaging.

    While the beads are centered on `center_slice`, any z-dependent XY wobble
    should remain in the extracted PSF.

    Parameters
    ----------
    data_3d : ndarray, shape (nx, ny, nz)
    locs : list of (xp, yp) - pixel positions of tagged beads (may be float)
    center_slice : int
        Z-slice index used for lateral centroid estimation.
    roi_half_xy : int
        Half-size of the output ROI in x and y (pixels).
    blur : sequence of 3 floats
        Gaussian sigma in (x, y, z) applied after averaging.

    Returns
    -------
    psf : ndarray, shape (2*roi_half_xy+1, 2*roi_half_xy+1, nz), max-normalised
    """
    import numpy as np
    import scipy.ndimage

    data_3d = data_3d.astype('f')
    n_x, n_y, n_z = data_3d.shape
    rs = roi_half_xy
    height = width = 2 * rs + 1

    # Fourier frequency grids for the ROI
    kx = np.fft.fftshift(np.arange(height) - rs) / height
    ky = np.fft.fftshift(np.arange(width) - rs) / width
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    psf_sum = np.zeros((height, width, n_z), dtype='f')

    for xp, yp in locs:
        xp_int = int(round(xp))
        yp_int = int(round(yp))

        # ROI is guaranteed in-bounds - no padding needed
        x0, x1 = xp_int - rs, xp_int + rs + 1
        y0, y1 = yp_int - rs, yp_int + rs + 1
        roi = data_3d[x0:x1, y0:y1, :].copy()

        # XY centroid from center slice only
        dx, dy = _xy_centroid(roi[:, :, center_slice])

        # Single phase ramp applied to the whole stack – preserves z-wobble
        phase = np.exp(-2j * np.pi * (KX * (-dx) + KY * (-dy)))
        for z_idx in range(n_z):
            psf_sum[:, :, z_idx] += np.fft.ifft2(
                np.fft.fft2(roi[:, :, z_idx]) * phase).real

    psf = psf_sum / len(locs)
    psf = scipy.ndimage.gaussian_filter(psf, blur)
    psf -= psf.min()
    if psf.max() > 1e-6:
        psf /= psf.max()
    return psf


class DHCalibrator(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        self.use_web_view = False

        self._calibrations = []
        self._optimizations = []

        # State for PSF extraction panel
        self._psf_locs = []           # list of (xp, yp) tagged bead positions
        self._psf_roi_half_xy = 30    # kept in sync with the panel text control

        dsviewer.paneHooks.append(self.GenDHPSFPanel)
        dsviewer.view.add_overlay(self.DrawOverlays, 'DH PSF ROIs')

        logging.debug('Adding menu items for double-helix PSF calibration')
        dsviewer.AddMenuItem('Processing', 'Calibrate DH PSF', self.OnCalibrate)
        dsviewer.AddMenuItem('Processing', 'Optimize DH PSF Detection', self.OnOptimizeDetection)
        dsviewer.AddMenuItem('Processing', 'Test DH PSF Detection', self.OnTestDetection)
    

    def GenDHPSFPanel(self, _pnl):
        import PYME.ui.manualFoldPanel as afp

        item = afp.foldingPane(_pnl, -1, caption='DH PSF Extraction', pinned=True)
        pan = wx.Panel(item, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        # Tag / Clear row
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        bTag = wx.Button(pan, -1, 'Tag Bead', style=wx.BU_EXACTFIT)
        bTag.Bind(wx.EVT_BUTTON, self.OnTagBead)
        hsizer.Add(bTag, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        bClear = wx.Button(pan, -1, 'Clear', style=wx.BU_EXACTFIT)
        bClear.Bind(wx.EVT_BUTTON, self.OnClearBeads)
        hsizer.Add(bClear, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.ALL, 0)

        # ROI half size
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'ROI half size [px]:'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.tDHPSFROI = wx.TextCtrl(pan, -1, value='30', size=(50, -1))
        self.tDHPSFROI.Bind(wx.EVT_TEXT, self.OnDHROIChanged)
        hsizer.Add(self.tDHPSFROI, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.EXPAND | wx.ALL, 0)

        # Center slice
        n_z = self.dsviewer.image.data_xyztc.shape[2]
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Center slice:'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.tDHCenterSlice = wx.SpinCtrl(pan, -1, value=str(n_z // 2),
                                          min=0, max=max(0, n_z - 1), size=(60, -1))
        hsizer.Add(self.tDHCenterSlice, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.EXPAND | wx.ALL, 0)

        # Blur
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, u'Blur \u03c3 (x,y,z):'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.tDHPSFBlur = wx.TextCtrl(pan, -1, value='0.5,0.5,1.0', size=(80, -1))
        hsizer.Add(self.tDHPSFBlur, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.EXPAND | wx.ALL, 0)

        # Extract button
        bExtract = wx.Button(pan, -1, 'Extract PSF', style=wx.BU_EXACTFIT)
        bExtract.Bind(wx.EVT_BUTTON, self.OnExtractDHPSF)
        vsizer.Add(bExtract, 0, wx.ALL | wx.ALIGN_RIGHT, 5)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)
        item.AddNewElement(pan)
        _pnl.AddPane(item)

    def OnTagBead(self, event):
        """Tag the bead nearest the cursor, or un-tag it if already tagged.

        Tags are rejected if the ROI would extend beyond the image boundary.
        """
        import numpy as np

        xp = self.dsviewer.do.xp
        yp = self.dsviewer.do.yp
        rs = self._psf_roi_half_xy

        # Un-tag if clicking within the ROI of an existing tag
        for i, (tx, ty) in enumerate(self._psf_locs):
            if (xp - tx) ** 2 + (yp - ty) ** 2 < rs ** 2:
                self._psf_locs.pop(i)
                self.dsviewer.view.Refresh()
                return

        # Reject if the ROI would extend beyond the image boundary
        image = self.dsviewer.image
        n_x, n_y = image.data_xyztc.shape[:2]
        xp_int, yp_int = int(round(xp)), int(round(yp))
        if (xp_int - rs < 0 or xp_int + rs + 1 > n_x or
                yp_int - rs < 0 or yp_int + rs + 1 > n_y):
            wx.MessageBox(
                f'Bead too close to the image edge for ROI half-size {rs} px.',
                'Tag Bead', wx.OK | wx.ICON_WARNING)
            return

        # Refine position: intensity-weighted centroid in center slice
        center_slice = self.tDHCenterSlice.GetValue()
        x0, x1 = xp_int - rs, xp_int + rs + 1
        y0, y1 = yp_int - rs, yp_int + rs + 1
        roi_2d = np.array(image.data_xyztc[x0:x1, y0:y1, center_slice, 0, 0]).squeeze()
        dx, dy = _xy_centroid(roi_2d)

        # dx/dy are offsets from the ROI centre; convert to image coordinates
        roi_cx = (x0 + x1 - 1) / 2.0
        roi_cy = (y0 + y1 - 1) / 2.0
        self._psf_locs.append((roi_cx + dx, roi_cy + dy))
        self.dsviewer.view.Refresh()

    def OnClearBeads(self, event):
        self._psf_locs.clear()
        self.dsviewer.view.Refresh()

    def OnDHROIChanged(self, event):
        try:
            self._psf_roi_half_xy = int(self.tDHPSFROI.GetValue())
        except ValueError:
            pass

    def DrawOverlays(self, view, dc):
        """Draw green ROI boxes around tagged bead positions (XY view only)."""
        if not self._psf_locs:
            return
        rs = self._psf_roi_half_xy
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('GREEN'), 1))
        if view.do.slice == view.do.SLICE_XY:
            for xp, yp in self._psf_locs:
                xp0, yp0 = view.pixel_to_screen_coordinates(xp - rs, yp - rs)
                xp1, yp1 = view.pixel_to_screen_coordinates(xp + rs, yp + rs)
                dc.DrawRectangle(int(xp0), int(yp0), int(xp1 - xp0), int(yp1 - yp0))

    def OnExtractDHPSF(self, wx_event=None):
        """Average XY-centered sub-stacks from all tagged bead positions."""
        import numpy as np
        from PYME.DSView.dsviewer import ImageStack, ViewIm3D
        from PYME.IO.MetaDataHandler import NestedClassMDHandler

        if not self._psf_locs:
            wx.MessageBox('No beads tagged.  Use "Tag Bead" first.',
                          'DH PSF Extraction', wx.OK | wx.ICON_WARNING)
            return

        try:
            roi_half_xy = int(self.tDHPSFROI.GetValue())
            blur = [float(s) for s in self.tDHPSFBlur.GetValue().split(',')]
        except ValueError as exc:
            wx.MessageBox(f'Invalid parameter: {exc}',
                          'DH PSF Extraction', wx.OK | wx.ICON_ERROR)
            return

        center_slice = self.tDHCenterSlice.GetValue()
        image = self.dsviewer.image
        n_z = image.data_xyztc.shape[2]
        n_c = image.data_xyztc.shape[4]

        psfs = []
        for chan in range(n_c):
            data = np.array(image.data_xyztc[:, :, :, 0, chan]).squeeze()
            psfs.append(_extract_dh_psf(data, self._psf_locs, center_slice, roi_half_xy, blur))

        psf_data = psfs[0] if n_c == 1 else psfs

        mdh = NestedClassMDHandler(image.mdh)
        mdh['ImageType'] = 'PSF'
        mdh['PSFExtraction.Mode'] = 'DH-XYonly'
        mdh['PSFExtraction.ROI'] = [roi_half_xy, roi_half_xy, n_z]
        mdh['PSFExtraction.Blur'] = blur
        mdh['PSFExtraction.CenterSlice'] = center_slice  # should remove this, use .Locations
        mdh['PSFExtraction.Locations'] = self._psf_locs

        im = ImageStack(data=psf_data, mdh=mdh, titleStub='Extracted DH PSF')
        im.defaultExt = '*.h5'
        ViewIm3D(im, mode='psf', parent=wx.GetTopLevelParent(self.dsviewer))
    

    def OnCalibrate(self, wx_event=None):
        from PYME.IO.FileUtils import nameUtils
        import matplotlib.pyplot as plt
        import matplotlib.cm
        import json
        import os
        import numpy as np
        from double_helix.z_mapping import calibrate_double_helix_psf
        from double_helix.z_range_dialog import ZRangeDialog
        from double_helix.detection_params_dialog import DetectionParamsDialog
        
        # query user for type of calibration
        # ftypes = ['Double Helix Theta', 'Double Helix Separate Gaussians']  # , 'AstigGaussGPUFitFR']
        # fit_type_dlg = wx.SingleChoiceDialog(self.dsviewer, 'Fit-type selection', 'Fit-type selection', ftypes)
        # fit_type_dlg.ShowModal()
        # fit_mod = ftypes[fit_type_dlg.GetSelection()]

        fit_mod = 'double_helix.DoubleGaussFit'

        # create dialog for user to input detection parameters
        # detection parameters then passed to calibrate_double_helix_psf
        detection_params_dialog = DetectionParamsDialog(None, defaultVal=[10, 1025, 215])
        succ = detection_params_dialog.ShowModal()
        if succ == wx.ID_OK:
            roi_half_size = int(detection_params_dialog.roi_half_size.GetValue())
            lobe_sep_guess = float(detection_params_dialog.lobe_sep_guess.GetValue())
            lobe_sigma_guess = float(detection_params_dialog.lobe_sigma_guess.GetValue())    

        results = calibrate_double_helix_psf(self.dsviewer.image, fit_mod, roi_half_size=roi_half_size, lobe_sep_guess=lobe_sep_guess, lobe_sigma_guess=lobe_sigma_guess)

        # do plotting
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
        
        zr_dialog = ZRangeDialog(None, minVal=results[0]['z'][0], maxVal=results[0]['z'][-1])
        succ = zr_dialog.ShowModal()
        if succ == wx.ID_OK:
            zmin = zr_dialog.zMin.GetValue()
            zmax = zr_dialog.zMax.GetValue()
        
        # write specified z range into results
        for ind, res in enumerate(results):
            res['z_range'] = (zmin, zmax)

        self._calibrations.append(results)

        fdialog = wx.FileDialog(None, 'Save Double Helix Calibration as ...',
            wildcard='dh_json (*.dh_json)|*.dh_json', style=wx.FD_SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath())  #, defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            fid = open(fpath, 'w', encoding='utf8')
            json.dump(results, fid, indent=4, sort_keys=True)
            fid.close()
            if self.use_web_view:  # save the html too
                fpath = os.path.splitext(fpath)[0] + '.html'
                with open(fpath, 'wb') as fid:
                    fid.write(html.encode('utf-8'))
            else:
                plt.savefig(os.path.splitext(fpath)[0] + '.png')
    
    def OnOptimizeDetection(self, wx_event=None):
        """Tests the steerable filtering on each slice of the input image and plots the results.
        """
        from double_helix.recipes.DH_mappings import OptimizeFilterSigma

        mod = OptimizeFilterSigma(input_image='input',
                                  output_plot='filter_sigma_plot', 
                                  output_data='max_strength')
        
        if mod.configure_traits(kind='modal'):
            namespace = {mod.input_image : self.dsviewer.image}
            mod.execute(namespace)
            self._optimizations.append({'output_plot': namespace[mod.output_plot], 
                                        'output_data': namespace[mod.output_data]})
            namespace[mod.output_plot].show()
    
    def OnTestDetection(self, wx_event=None):
        from PYME.DSView import ViewIm3D
        from PYME.DSView import overlays
        from PYME.IO import tabular
        from double_helix.recipes.DH_mappings import DetectDoubleHelices

        mod = DetectDoubleHelices()
        if mod.configure_traits(kind='modal'):
            namespace = {mod.input_image : self.dsviewer.image}
            mod.execute(namespace)

            #Open the result in a new window. 
            dsv = ViewIm3D(namespace[mod.output_strength_im], parent=self.dsviewer, glCanvas=self.dsviewer.glCanvas,
                     title=mod.output_strength_im)
            
            # add detections as overlays
            filt = tabular.MappingFilter(namespace[mod.output_detections])

            dsv._ovl = overlays.PointDisplayOverlay(filter=filt, display_name='Detections')
            dsv._ovl.pointMode = 'lm'
            z_mode = 't' if self.dsviewer.image.data_xyztc.shape[2] < self.dsviewer.image.data_xyztc.shape[3] else 'z'
            dsv._ovl.z_mode = z_mode
            dsv._ovl.pointSize = mod.fit_roi_half_size * 2 + 1
            dsv.view.add_overlay(dsv._ovl)

            if not hasattr(self.dsviewer, '_ovl') or not hasattr(self.dsviewer._ovl, 'filter'):
                self.dsviewer._ovl = overlays.PointDisplayOverlay(filter=filt, display_name='Detections')
                self.dsviewer._ovl.pointMode = 'lm'
                self.dsviewer._ovl.z_mode = z_mode
                self.dsviewer._ovl.pointSize = mod.fit_roi_half_size * 2 + 1
                self.dsviewer.view.add_overlay(self.dsviewer._ovl)
            else:
                self.dsviewer._ovl.filter.setResults(filt)


def Plug(dsviewer):
    dsviewer.DH_tools = DHCalibrator(dsviewer)



