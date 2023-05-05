
import numpy as np
from PYME.IO import MetaDataHandler
md = MetaDataHandler.SimpleMDHandler()

md['StartTime'] = 1300676151.901
md['tIndex'] = 0
md['Analysis.BGRange'] = [0, 0]
# md['Analysis.DataFileID'] = 1571469165
# md['Analysis.DebounceRadius'] = 14
md['Analysis.DetectionThreshold'] = 0.75E6
md['Analysis.FitModule'] = 'DoubleHelixFit_Theta'
# md['Analysis.InterpModule'] = 'CSInterpolator'
#md['Analysis.EstimatorModule'] = 'priEstimator'
md['Analysis.subtractBackground'] = False
md['Analysis.ROISize'] = 10
md['Camera.ADOffset'] = 110
md['Camera.CycleTime'] = 0.25178998708724976
md['Camera.EMGain'] = 272
md['Camera.ElectronsPerCount'] = 12.9
md['Camera.IntegrationTime'] = 0.25
md['Camera.Name'] = 'Andor IXon DV97'
md['Camera.NoiseFactor'] = 1.4
md['Camera.ReadNoise'] = 1
md['Camera.TrueEMGain'] = 33.415239495686144
md['voxelsize.units'] = 'um'
md['voxelsize.x'] = 0.1177
md['voxelsize.y'] = 0.1177
md['voxelsize.z'] = 0.1
md['Analysis.SigmaGuess'] = 200
md['Analysis.LobeSepGuess'] = 900
md['Analysis.DetectionFilterMag'] = 0.15

# A0, A1, x, y, theta, lobe_sep, sig, bg = p
md['Test.DefaultParams'] = [2000, 2000, 0, 0, 0, 900, 200, 50]
md['Test.ParamJitter'] = [200, 200, 117, 117, np.pi, 100, 25, 25]
# md['Test.SimModule'] = u'InterpFitR'
md['Test.ROISize'] = md['Analysis.ROISize']


def test_DoubleHelixFit_Theta():
    """Test the Astigmatic Gaussian fit by fitting some randomly generated events. The pass condition here is fairly
    loose, but should be sufficient to detect when the code has been broken"""
    from PYME.localization.Test import fitTestJigWC as fitTestJig

    tj = fitTestJig.fitTestJig(md, 'DoubleHelixFit_Theta')
    tj.runTests(nTests=100)

    errors_over_pred_IQR = fitTestJig.IQR((tj.error('x0') / tj.res['fitError']['x0']))
    print(errors_over_pred_IQR)
    
    assert errors_over_pred_IQR < 3
