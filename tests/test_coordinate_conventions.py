
import numpy as np
from PYME.IO.image import ImageStack
from PYME.IO import MetaDataHandler
from double_helix import DoubleGaussFit

# assume theta ranges from 0 to pi, and z from -1000 to 1000 nm
step_size = 60  # [nm]
_z = np.arange(-1000, 1000.1, step_size)  # [nm]
_z_um = _z / 1e3  # [um]
_theta = np.linspace(0, 0.99 * np.pi, len(_z))  # [rad]


md = MetaDataHandler.SimpleMDHandler()
md['Analysis.ROISize'] = 10  # actually 0.5 pix less than half size.
md['voxelsize.units'] = 'um'
md['voxelsize.x'] = 0.12
md['voxelsize.y'] = 0.12
md['voxelsize.z'] = step_size / 1e3  # [um]
md['Analysis.SigmaGuess'] = 200  # Double Helix Lobe Sigma Guess [nm]
md['Analysis.LobeSepGuess'] = 1050  # Double Helix Lobe
md['Analysis.DetectionFilterSigma'] = 5  # Detection Filter Sigma (in px)
md['Origin.z'] = _z_um[0]  # set Z origin to match the first Z position in the stack
A0 = 500
# simulate an image stack to match this calibration
psf_stack_arr = np.empty((2 * md['Analysis.ROISize'] + 1, 2 * md['Analysis.ROISize'] + 1, len(_z)), float)
for zind in range(len(_z)):
    psf_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
        [A0, A0, 0, 0, _theta[zind], md['Analysis.LobeSepGuess'], md['Analysis.SigmaGuess'], 20], 
        md, x=md['voxelsize.x']*md['Analysis.ROISize'], y=md['voxelsize.y']*md['Analysis.ROISize'], roiHalfSize=md['Analysis.ROISize']
    )



# # simulate a Z stepped acquisition, where the focus is stepped in Z while the sample is fixed,
# # so theta unwinds as Z increases
step_size_ind = 5
focus_step_z = step_size_ind * step_size  # [nm]
# make sure that we don't wrap theta by subtracting off too much. Clip the whole thing by step_size_ind indices
trimmed_theta = _theta[step_size_ind:-step_size_ind]
trimmed_z = _z[step_size_ind:-step_size_ind]
focus_step_theta = trimmed_theta[step_size_ind] - trimmed_theta[0]  # [rad]
focus = np.hstack([np.zeros_like(trimmed_z), step_size_ind * step_size * np.ones_like(trimmed_z)])  # [nm])
acquired_theta = np.hstack([trimmed_theta, trimmed_theta - focus_step_theta])  # [rad]


test_stack_arr = np.empty((2 * md['Analysis.ROISize'] + 1, 2 * md['Analysis.ROISize'] + 1, len(acquired_theta)), float)
for zind in range(len(acquired_theta)):
    test_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
        [A0, A0, 0, 0, acquired_theta[zind], md['Analysis.LobeSepGuess'], md['Analysis.SigmaGuess'], 20], 
        md, x=md['voxelsize.x']*md['Analysis.ROISize'], y=md['voxelsize.y']*md['Analysis.ROISize'], roiHalfSize=md['Analysis.ROISize']
    )
test_stack = ImageStack(test_stack_arr, md)
from PYME.IO.tabular import DictSource
gt_dict = {
    'z': np.hstack([trimmed_z, trimmed_z + focus_step_z]), 
    'x': np.zeros_like(focus), 'y': np.zeros_like(focus), 
    'fitResults_theta': acquired_theta, 'fitError_theta': 0.001 * np.ones_like(acquired_theta),
    'fitResults_sigma': md['Analysis.SigmaGuess'] * np.ones_like(acquired_theta),
    'fitResults_lobesep': md['Analysis.LobeSepGuess'] * np.ones_like(acquired_theta),
    'fitResults_A0': A0 * np.ones_like(acquired_theta),
    'fitResults_A1': A0 * np.ones_like(acquired_theta),
    'startParams_x0': np.zeros_like(acquired_theta),
    'startParams_y0': np.zeros_like(acquired_theta),
    'fitError_sigma': np.ones_like(acquired_theta),
    'fitError_lobesep': np.ones_like(acquired_theta),
}
gt = DictSource(gt_dict)
gt.mdh = MetaDataHandler.DictMDHandler(md)
gt_focus = gt_dict.copy()
gt_focus['z'] = focus
gt_focus = DictSource(gt_focus)
gt_focus.mdh = MetaDataHandler.DictMDHandler(md)

def test_calibration_z():
    from double_helix.z_mapping import calibrate_double_helix_psf
    # NOTE - PYME has a transpose between localizations and images
    psf_stack = ImageStack(psf_stack_arr, md, haveGUI=False)
    cal = calibrate_double_helix_psf(psf_stack, 'double_helix.DoubleGaussFit', roi_half_size=md['Analysis.ROISize'], filter_sigma=md['Analysis.DetectionFilterSigma'], lobe_sep_guess=md['Analysis.LobeSepGuess'], lobe_sigma_guess=md['Analysis.SigmaGuess'])
    c_ind = 0  # which channel to test (if multiple channels in calibration)

    unwrapped_theta_gt = np.unwrap(_theta, np.pi/2, period=np.pi)
    unwrapped_theta_cal = np.unwrap(np.asarray(cal[c_ind]['theta']), np.pi/2, period=np.pi)

    # check that Z positions in calibration match expected Z positions (which are just _z)
    np.testing.assert_almost_equal(_z, cal[c_ind]['z'], decimal=1)
    
    # import matplotlib.pyplot as plt
    # plt.plot(_z, _theta, label='gt', linewidth=5)
    # plt.plot(_z, unwrapped_theta_gt, label='gt unwrapped')
    # plt.plot(cal[c_ind]['z'], unwrapped_theta_cal, label='cal unwrapped')
    # plt.legend()
    # plt.show()
    # # theta_spline = LSQUnivariateSpline(z_valid, unwrapped_theta_cal, knots, ext='const')
    # # theta_cal = theta_spline(z_v)

    np.testing.assert_almost_equal(unwrapped_theta_cal, unwrapped_theta_gt, decimal=1)  # check that theta vs z relationship in calibration matches expected theta vs z relationship (which is just _theta vs _z)


def test_z_focus_mapping():
    from double_helix.z_mapping import calibrate_double_helix_psf
    # NOTE - PYME has a transpose between localizations and images
    psf_stack = ImageStack(psf_stack_arr, md, haveGUI=False)
    cal = calibrate_double_helix_psf(psf_stack, 'double_helix.DoubleGaussFit', roi_half_size=md['Analysis.ROISize'], filter_sigma=md['Analysis.DetectionFilterSigma'], lobe_sep_guess=md['Analysis.LobeSepGuess'], lobe_sigma_guess=md['Analysis.SigmaGuess'])
    c_ind = 0  # which channel to test (if multiple channels in calibration)

    # look up the Z position based on the "acquired_theta" values, and check that it matches the expected Z positions (which are just _z shifted by the focus step)
    from double_helix.recipes.DH_mappings import DoubleHelixMapZ

    # save cal to temp location to test loading from file in mapping function
    import json
    import tempfile
    import os

    # add z_range to each channel's calibration (required by lookup_dh_z)
    for chan_cal in cal:
        chan_cal['z_range'] = (_z[0] - 1, _z[-1] + 1)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dh_json', delete=False, encoding='utf8') as f:
        temp_path = f.name
        json.dump(cal, f)
    # file must be closed before unifiedIO.read can open it on Windows
    try:
        # KEY POINT:
        # "z" input to DoubleHelixMapZ is the "focus" Z, i.e. obj position.
        recipe_results = DoubleHelixMapZ(calibration_location=temp_path,
                                 target_knot_spacing=2 * step_size + 1).apply(input_name=gt_focus)  # ['dh_localizations']
        # "z" output is "dh_z + z_input"
    finally:
        os.unlink(temp_path)
    mapped = recipe_results['dh_localizations']
    plot = recipe_results['dh_z_lookup_plot']

    # mapping Z with DH should recover the original Z positions, with no focus step
    np.testing.assert_allclose(mapped['z'], np.hstack([trimmed_z, trimmed_z]), atol=1.1)

