
import numpy
from PYME.DSView.modules._base import Plugin
import wx
import logging
logger = logging.getLogger(__name__)

class DHCalibrator(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        #generate new tab to show results
        self.use_web_view = True
        if not '_dh_view' in dir(self):
            try:
                self._dh_view= wx.html2.WebView.New(self.dsviewer)
                self.dsviewer.AddPage(self._dh_view, True, 'Double-Helix Cal.')

            except (NotImplementedError, AttributeError):
                self.use_web_view = False
        
        dsviewer.AddMenuItem('Processing', 'Calibrate DH PSF', self.OnCalibrate)        
        # dsviewer.AddMenuItem('Processing', 'Generate &Mask', self.OnApplyThreshold)
        # dsviewer.AddMenuItem('Processing', '&Label', self.OnLabelSizeThreshold)   

    def OnCalibrate(self, wx_event=None):
        from PYME.IO.FileUtils import nameUtils
        import matplotlib.pyplot as plt
        import matplotlib.cm
        import json
        from double_helix.z_mapping import calibrate_double_helix_psf
        # query user for type of calibration
        # ftypes = ['Double Helix Theta', 'Double Helix Separate Gaussians']  # , 'AstigGaussGPUFitFR']
        # fit_type_dlg = wx.SingleChoiceDialog(self.dsviewer, 'Fit-type selection', 'Fit-type selection', ftypes)
        # fit_type_dlg.ShowModal()
        # fit_mod = ftypes[fit_type_dlg.GetSelection()]
        fit_mod = 'DoubleHelixFit_Theta'

        res = calibrate_double_helix_psf(self.dsviewer.image, fit_mod)

        # do plotting
        plt.ioff()
        f = plt.figure(figsize=(10, 4))
        # colors = iter(matplotlib.cm.Dark2(np.linspace(0, 1, 2*self.image.data.shape[3])))
        plt.subplot(121)
        for i, res in enumerate(res):
            # nextColor1 = next(colors)
            # nextColor2 = next(colors)
            if 'heta' in fit_mod:
                plt.plot(res['z'], res['theta'], label='Theta [rad.]')
                plt.plot(res['z'], res['lobesep'], label='Lobe Separation [nm]')
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

        fdialog = wx.FileDialog(None, 'Save Double Helix Calibration as ...',
            wildcard='dh_json (*.dh_json)|*.dh_json', style=wx.FD_SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath())  #, defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            fid = open(fpath, 'w', encoding='utf8')
            json.dump(res, fid, indent=4, sort_keys=True)
            fid.close()
            # if use_web_view:  # save the html too
            #     import os
            #     fpath = os.path.splitext(fpath)[0] + '.html'
            #     with open(fpath, 'wb') as fid:
            #         fid.write(html.encode('utf-8'))

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



