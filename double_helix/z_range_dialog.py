
import wx

class ZRangeDialog(wx.Dialog):
    def __init__(self, parent, minVal=-1e3, maxVal=1e3):
        wx.Dialog.__init__(self, parent)

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
   
        sizer2.Add(wx.StaticText(self, -1, u'Min Z  [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.zMin = wx.TextCtrl(self, -1, '%1.6G' % minVal, size=(80, -1))
        

        sizer2.Add(self.zMin, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(wx.StaticText(self, -1, u'Max Z [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.zMax = wx.TextCtrl(self, -1, '%1.6G' % maxVal, size=(80, -1))

        sizer2.Add(self.zMax, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALL, 5)    
        
        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)