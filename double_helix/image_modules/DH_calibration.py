
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
        
        logging.debug('Adding menu items for double-helix PSF calibration')
        dsviewer.AddMenuItem('Processing', 'Calibrate DH PSF', self.OnCalibrate)        
        # dsviewer.AddMenuItem('Processing', 'Generate &Mask', self.OnApplyThreshold)
        # dsviewer.AddMenuItem('Processing', '&Label', self.OnLabelSizeThreshold)   

    def OnCalibrate(self, wx_event=None):
        from PYME.IO.FileUtils import nameUtils
        import matplotlib.pyplot as plt
        import matplotlib.cm
        import json
        import os
        from double_helix.z_mapping import calibrate_double_helix_psf
        from double_helix.z_range_dialog import ZRangeDialog
        from double_helix.z_mapping import OptimalSigmabyFrame
        from double_helix.z_mapping import StrengthVsZ
        from double_helix.z_mapping import StrengthVsSigma
        # query user for type of calibration
        # ftypes = ['Double Helix Theta', 'Double Helix Separate Gaussians']  # , 'AstigGaussGPUFitFR']
        # fit_type_dlg = wx.SingleChoiceDialog(self.dsviewer, 'Fit-type selection', 'Fit-type selection', ftypes)
        # fit_type_dlg.ShowModal()
        # fit_mod = ftypes[fit_type_dlg.GetSelection()]
        fit_mod = 'double_helix.DoubleGaussFit'

        results = calibrate_double_helix_psf(self.dsviewer.image, fit_mod)

        # determine optimal filter sigma for middle slice of z stack
        # initial sigma search from 0.2 - 20 px in 0.2 px increments 
        dh_stack=numpy.squeeze(self.dsviewer.image.data_xytc[:,:,:,0]) 
        mid_slice = int(dh_stack.shape[2]/2)
        sigmaUB = 20 # filter sigma upperbound for filter sigma sweep for each slice
        initial_sigs = numpy.array(range(20,sigmaUB*100,20), dtype=float)/100
        midSlice_filterSigma, midSlice_strengths = StrengthVsSigma(initial_sigs, dh_stack[:,:,mid_slice])
        
        
        # determine optimal detection filter sigma for each slice of z stack
        # for each slice of z stack, run detection on psf for each sigma in sigs
        # save filter sigma producing highest detector response for that slice
        # sigma values for search range from +/- 3 px from optimal sigma for middle slice, 0.2 px increments
        sigs = numpy.array(range((midSlice_filterSigma-3)*100,(midSlice_filterSigma+3)*100,20), dtype=float)/100
        optimalSigs, maxStrengths, maxXforOptSigma, maxYforOptSigma = OptimalSigmabyFrame(dh_stack, sigs)
        
        # determine how well filter sigma optimized for middle slice detects across PSFs across z stack
        # for each slice of z stack, run detection with filter sigma optimal for middle slice
        #   (normalization factor computed with middle slice lobesep and lobe sigma from calibration)
        # plot detection strength at each slice and detection strength normalized for dh amplitude at each slice
        px_size_nm = self.dsviewer.image.mdh['voxelsize.x']*1000
        mid_slice_filter_sigma = optimalSigs[mid_slice]
        mid_slice_lobesep_px = results[0]['lobesep'][mid_slice]/px_size_nm
        mid_slice_sigma_px = results[0]['sigma'][mid_slice]/px_size_nm
        strengths, dhAmplitudes, maxX, maxY = StrengthVsZ(dh_stack, sigma=mid_slice_filter_sigma, l=mid_slice_lobesep_px, s=mid_slice_sigma_px)

        fig = plt.figure(figsize=(12, 10))
        gspec = fig.add_gridspec(nrows=6, ncols=2)
        ax1 = fig.add_subplot(gspec[0:2, 0])
        ax2 = fig.add_subplot(gspec[3:5, 0])
        ax3 = fig.add_subplot(gspec[0:1, 1])
        ax4 = fig.add_subplot(gspec[2:3, 1])
        ax5 = fig.add_subplot(gspec[4:5, 1])

        ax1.plot(initial_sigs, midSlice_strengths)
        ax1.set_xlabel("Filter Sigma (px)")
        ax1.set_ylabel("Raw Detection Strength")
        ax1.title.set_text('Middle Slice Detection Strength vs Filter Sigma')
        
        ax2.plot(results[0]['z'], optimalSigs)
        ax2.set_xlabel("z (nm)")
        ax2.set_ylabel("Optimal Filter Sigma")
        ax2.title.set_text('Optimal Filter Sigma vs Z')

        ax3.plot(results[0]['z'], dhAmplitudes)
        ax3.set_xlabel("z (nm)")
        ax3.set_ylabel("DH Amplitude")
            
        ax4.plot(results[0]['z'], strengths)
        ax4.set_xlabel("z (nm)")
        ax4.set_ylabel("Max Strength")
            
        ax5.plot(results[0]['z'], strengths/dhAmplitudes)
        ax5.set_xlabel("z (nm)")
        ax5.set_ylabel("Max Strength/DH Amplitude")

        # do plotting
        plt.ioff()
        f = plt.figure(figsize=(10, 12))
        # colors = iter(matplotlib.cm.Dark2(np.linspace(0, 1, 2*self.image.data.shape[3])))
        plt.subplot(311)
        for ind, res in enumerate(results):
            # nextColor1 = next(colors)
            # nextColor2 = next(colors)
            if 'heta' in fit_mod:
                plt.plot(res['z'], res['theta'], label='Theta [rad.]')
                plt.legend()
                plt.subplot(312)
                plt.plot(res['z'], res['lobesep'], label='Lobe Separation [nm]')
                plt.legend()
            plt.subplot(313)
            plt.plot(res['z'], res['sigma'], label='sigma [nm]')
            
            # lbz = np.absolute(res['z'] - res['zRange'][0]).argmin()
            # ubz = np.absolute(res['z'] - res['zRange'][1]).argmin()
            # plt.plot(res['z'], res['sigmax'], ':', c=nextColor1)  # , label='x - %d' % i)
            # plt.plot(res['z'], res['sigmay'], ':', c=nextColor2)  # , label='y - %d' % i)
            # plt.plot(res['z'][lbz:ubz], res['sigmax'][lbz:ubz], label='x - %d' % i, c=nextColor1)
            # plt.plot(res['z'][lbz:ubz], res['sigmay'][lbz:ubz], label='y - %d' % i, c=nextColor2)

        #plt.ylim(-200, 400)
        plt.grid()
        plt.xlabel('z position [nm]')
        # plt.ylabel('')
        plt.legend()

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

        plt.tight_layout()

        plt.ion()
        # #dat = {'z' : objPositions['z'][valid].tolist(), 'sigmax' : res['fitResults_sigmax'][valid].tolist(),
        # #                   'sigmay' : res['fitResults_sigmay'][valid].tolist(), 'dsigma' : dsigma[valid].tolist()}


        if self.use_web_view:
            # fig = mpld3.fig_to_html(f)
            # data = json.dumps(results)

            # # template = env.get_template('astigCal.html')
            # html = template.render(astigplot=fig, data=data)
            # #print html
            # self._astig_view.SetPage(html, '')
            plt.show()
        else:
            plt.show()
        
        zr_dialog = ZRangeDialog(None)
        succ = zr_dialog.ShowModal()
        if succ == wx.ID_OK:
            zmin = zr_dialog.zMin.GetValue()
            zmax = zr_dialog.zMax.GetValue()
        
        # write specified z range into results
        for ind, res in enumerate(results):
            res['z_range'] = (zmin, zmax)

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

def Plug(dsviewer):
    dsviewer.PSFTools = DHCalibrator(dsviewer)
    # if dsviewer.do.ds.shape[2] > 1:
    #     dsviewer.crbv = CRBViewPanel(dsviewer, dsviewer.image)
    #     dsviewer.dataChangeHooks.append(dsviewer.crbv.calcCRB)
        
    #     dsviewer.psfqp = PSFQualityPanel(dsviewer)
    #     dsviewer.dataChangeHooks.append(dsviewer.psfqp.FillGrid)
        
        #dsviewer.AddPage(dsviewer.psfqp, False, 'PSF Quality')
        # dsviewer.AddPage(dsviewer.crbv, False, 'Cramer-Rao Bounds')
            
        # pinfo1 = aui.AuiPaneInfo().Name("psfQPanel").Left().Caption('PSF Quality').DestroyOnClose(True).CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        # dsviewer._mgr.AddPane(dsviewer.psfqp, pinfo1)
        
    # dsviewer._mgr.Update()



