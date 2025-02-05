
import numpy
from PYME.DSView.modules._base import Plugin
import wx
import logging
logger = logging.getLogger(__name__)

class DHCalibrator(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        # #generate new tab to show results
        # self.use_web_view = True
        # if not '_dh_view' in dir(self):
        #     try:
        #         self._dh_view= wx.html2.WebView.New(self.dsviewer)
        #         self.dsviewer.AddPage(self._dh_view, True, 'Double-Helix Cal.')

        #     except (NotImplementedError, AttributeError):
        self.use_web_view = False

        self._calibrations = []
        self._optimizations = []
        
        logging.debug('Adding menu items for double-helix PSF calibration')
        dsviewer.AddMenuItem('Processing', 'Calibrate DH PSF', self.OnCalibrate)        
        dsviewer.AddMenuItem('Processing', 'Optimize DH PSF Detection', self.OnOptimizeDetection)

    def OnCalibrate(self, wx_event=None):
        from PYME.IO.FileUtils import nameUtils
        import matplotlib.pyplot as plt
        import matplotlib.cm
        import json
        import os
        from double_helix.z_mapping import calibrate_double_helix_psf
        from double_helix.z_range_dialog import ZRangeDialog
        
        # query user for type of calibration
        # ftypes = ['Double Helix Theta', 'Double Helix Separate Gaussians']  # , 'AstigGaussGPUFitFR']
        # fit_type_dlg = wx.SingleChoiceDialog(self.dsviewer, 'Fit-type selection', 'Fit-type selection', ftypes)
        # fit_type_dlg.ShowModal()
        # fit_mod = ftypes[fit_type_dlg.GetSelection()]
        fit_mod = 'double_helix.DoubleGaussFit'

        results = calibrate_double_helix_psf(self.dsviewer.image, fit_mod)

        # do plotting
        plt.ioff()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for ind, res in enumerate(results):
            axes[0].plot(res['z'], res['theta'], label='chan. %d' % ind)
            axes[0].set_ylabel('Theta [rad.]')

            axes[1].plot(res['z'], res['lobesep'], label='chan. %d' % ind)
            axes[1].set_ylabel('Lobe Separation [nm]')
            
            axes[2].plot(res['z'], res['sigma'], label='chan. %d' % ind)
            axes[2].set_ylabel('Sigma [nm]')

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

def Plug(dsviewer):
    dsviewer.DH_tools = DHCalibrator(dsviewer)



