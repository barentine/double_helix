import wx

class DetectionParamsDialog(wx.Dialog):
    def __init__(self, parent, defaultVal=-1e3):
        wx.Dialog.__init__(self, parent)

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(self, -1, "Enter DH PSF Detection Parameters")
        font = title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        main_sizer.Add(title, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        # ROI Half Size
        row1 = wx.BoxSizer(wx.HORIZONTAL)
        row1.Add(wx.StaticText(self, -1, u'ROI Half Size [px]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.roi_half_size = wx.TextCtrl(self, -1, '%1.6G' % defaultVal, size=(80, -1))
        row1.Add(self.roi_half_size, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        main_sizer.Add(row1, 0, wx.EXPAND | wx.ALL, 0)

        # Detection Filter Sigma
        row2 = wx.BoxSizer(wx.HORIZONTAL)
        row2.Add(wx.StaticText(self, -1, u'Detection Filter Sigma [px]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.detection_filter_sigma = wx.TextCtrl(self, -1,'%1.6G' % defaultVal, size=(80, -1))
        row2.Add(self.detection_filter_sigma, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        main_sizer.Add(row2, 0, wx.EXPAND | wx.ALL, 0)

        # Initial Lobe Sep. Guess
        row3 = wx.BoxSizer(wx.HORIZONTAL)
        row3.Add(wx.StaticText(self, -1, u'Initial Lobe Sep. Guess [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.lobe_sep_guess = wx.TextCtrl(self, -1, '%1.6G' % defaultVal, size=(80, -1))
        row3.Add(self.lobe_sep_guess, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        main_sizer.Add(row3, 0, wx.EXPAND | wx.ALL, 0)

        # Initial Lobe Sigma Guess
        row4 = wx.BoxSizer(wx.HORIZONTAL)
        row4.Add(wx.StaticText(self, -1, u'Initial Lobe Sigma Guess [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.lobe_sigma_guess = wx.TextCtrl(self, -1, '%1.6G' % defaultVal, size=(80, -1))
        row4.Add(self.lobe_sigma_guess, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        main_sizer.Add(row4, 0, wx.EXPAND | wx.ALL, 0)

        # Buttons
        btSizer = wx.StdDialogButtonSizer()
        btn_ok = wx.Button(self, wx.ID_OK)
        btn_ok.SetDefault()
        btSizer.AddButton(btn_ok)
        btn_cancel = wx.Button(self, wx.ID_CANCEL)
        btSizer.AddButton(btn_cancel)
        btSizer.Realize()
        main_sizer.Add(btSizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        self.SetSizer(main_sizer)
        main_sizer.Fit(self)
        