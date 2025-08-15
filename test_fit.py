
import numpy as np
from PYME.IO import MetaDataHandler
md = MetaDataHandler.SimpleMDHandler()

fit_mod = 'double_helix.DoubleGaussFit'

md['StartTime'] = 1300676151.901
md['tIndex'] = 0
md['Analysis.BGRange'] = [0, 0]

md['Analysis.DetectionThreshold'] = 1
md['Analysis.FitModule'] = fit_mod

md['Analysis.subtractBackground'] = False
md['Analysis.ROISize'] = 10
md['Camera.ADOffset'] = 178
md['Camera.CycleTime'] = 0.25178998708724976
md['Camera.EMGain'] = 300
md['Camera.ElectronsPerCount'] = 12.9
md['Camera.IntegrationTime'] = 0.25
md['Camera.Name'] = 'Andor IXon DV97'
md['Camera.NoiseFactor'] = 1.4
md['Camera.ReadNoise'] = 1
md['Camera.TrueEMGain'] = 284
md['voxelsize.units'] = 'um'
md['voxelsize.x'] = 0.12
md['voxelsize.y'] = 0.12
md['voxelsize.z'] = 0.1
md['Analysis.SigmaGuess'] = 200  # Double Helix Lobe Sigma Guess [nm]
md['Analysis.LobeSepGuess'] = 1050  # Double Helix Lobe Separation Guess [nm]
md['Analysis.DetectionFilterSigma'] = 5  # Detection Filter Sigma (in px)

# A0, A1, x, y, theta, lobe_sep, sig, bg = p
md['Test.DefaultParams'] = [500, 500, 0, 0, 0, md['Analysis.LobeSepGuess'], md['Analysis.SigmaGuess'], 20]
md['Test.ParamJitter'] = [50, 50, 120, 120, 0.5*np.pi, 20, 10, 10]
md['Test.ROISize'] = md['Analysis.ROISize']


def test_DoubleHelixFit_Theta():
    """flexible, but hopefully sufficient to detect when the code has been broken"""
    from PYME.localization.Test import fitTestJigWC as fitTestJig

    tj = fitTestJig.fitTestJig(md, fit_mod)
    tj.runTests(nTests=100)
    
    assert np.abs(tj.error('x0')).mean() < 0.1 * (md['voxelsize.x'] * 1e3)  # 0.1 [pix] in [nm]
    assert np.abs(tj.error('theta')).mean() < 0.1  # [rad.]
