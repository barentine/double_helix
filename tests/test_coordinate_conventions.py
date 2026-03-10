
import numpy as np
from PYME.IO.image import ImageStack
from PYME.IO import MetaDataHandler
from PYME.IO.tabular import DictSource
from double_helix import DoubleGaussFit

# ============================================================================
# Setup: Calibration PSF Stack
# ============================================================================
# Simulate a calibration with theta ranging from 0 to pi and z from -1000 to 1000 nm
step_size = 60  # [nm]
_z = np.arange(-1000, 1000.1, step_size)  # [nm]
_z_um = _z / 1e3  # [um]
_theta = np.linspace(0, 0.99 * np.pi, len(_z))  # [rad]

# Setup metadata for the calibration
md = MetaDataHandler.SimpleMDHandler()
md['Analysis.ROISize'] = 10  # actually 0.5 pix less than half size
md['voxelsize.units'] = 'um'
md['voxelsize.x'] = 0.12
md['voxelsize.y'] = 0.12
md['voxelsize.z'] = step_size / 1e3  # [um]
md['Analysis.SigmaGuess'] = 200  # Double Helix Lobe Sigma Guess [nm]
md['Analysis.LobeSepGuess'] = 1050  # Double Helix Lobe separation [nm]
md['Analysis.DetectionFilterSigma'] = 5  # Detection Filter Sigma [px]
md['Origin.z'] = _z_um[0]  # set Z origin to match the first Z position in the stack
A0 = 500  # amplitude

# Generate PSF stack: for each z position, create a synthetic double helix image
# with the corresponding theta
psf_stack_arr = np.empty(
    (2 * md['Analysis.ROISize'] + 1, 2 * md['Analysis.ROISize'] + 1, len(_z)), 
    float
)
for zind in range(len(_z)):
    psf_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
        [A0, A0, 0, 0, _theta[zind], md['Analysis.LobeSepGuess'], 
         md['Analysis.SigmaGuess'], 20],
        md, 
        x=md['voxelsize.x'] * md['Analysis.ROISize'],
        y=md['voxelsize.y'] * md['Analysis.ROISize'],
        roiHalfSize=md['Analysis.ROISize']
    )


# ============================================================================
# Setup: Test stack with focus step
# ============================================================================
# Simulate a real acquisition with two imaging sessions separated by a focus step
# Before focus step: sample at z positions [step_size_ind, ..., -step_size_ind]
# After focus step: same sample imaged again, but objective shifted by focus_step_z
# The double helix rotates with the focus step, so theta adjusts accordingly

step_size_ind = 5  # number of steps for the focus shift
focus_step_z = step_size_ind * step_size  # [nm] objective position change

# Trim indices to avoid theta wrapping complications (use middle portion of calibration)
trimmed_theta = _theta[step_size_ind:-step_size_ind]
trimmed_z = _z[step_size_ind:-step_size_ind]

# The focus step introduces a rotation in the double helix PSF
focus_step_theta = trimmed_theta[step_size_ind] - trimmed_theta[0]  # [rad]

# Create the focus positions: zero for first session, focus_step_z for second
focus_positions = np.hstack([
    np.zeros_like(trimmed_z),
    step_size_ind * step_size * np.ones_like(trimmed_z)
])

# Acquired theta: same structure imaged twice, but rotated on second pass 
acquired_theta = np.hstack([
    trimmed_theta,
    trimmed_theta - focus_step_theta  # theta decreases due to rotation
])

# Generate test image stack
test_stack_arr = np.empty(
    (2 * md['Analysis.ROISize'] + 1, 2 * md['Analysis.ROISize'] + 1, len(acquired_theta)),
    float
)
for zind in range(len(acquired_theta)):
    test_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
        [A0, A0, 0, 0, acquired_theta[zind], md['Analysis.LobeSepGuess'],
         md['Analysis.SigmaGuess'], 20],
        md,
        x=md['voxelsize.x'] * md['Analysis.ROISize'],
        y=md['voxelsize.y'] * md['Analysis.ROISize'],
        roiHalfSize=md['Analysis.ROISize']
    )
test_stack = ImageStack(test_stack_arr, md)

# Create ground truth data source with actual sample positions (as if focus shift had not occurred)
gt_dict = {
    'z': np.hstack([trimmed_z, trimmed_z]),  # actual sample z (same structure twice)
    'x': np.zeros_like(focus_positions),
    'y': np.zeros_like(focus_positions),
    'fitResults_theta': acquired_theta,  # what we actually observed
    'fitError_theta': 0.001 * np.ones_like(acquired_theta),
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

# Create a version with focus positions instead of actual sample positions
# (used for mapping - input to DoubleHelixMapZ is the objective/focus position)
gt_focus = gt_dict.copy()
gt_focus['z'] = focus_positions  # objective position, not sample position
gt_focus = DictSource(gt_focus)
gt_focus.mdh = MetaDataHandler.DictMDHandler(md)
# ============================================================================
# TEST 1: Calibration PSF Stack → Recovered Theta vs Z
# ============================================================================
def test_calibration_z():
    """
    Test that calibration from a PSF stack recovers the input theta vs z relationship.
    
    This test:
    1. Creates a calibration by fitting each PSF slice to extract theta at each z
    2. Checks that the recovered z positions match the input z positions
    3. Checks that the theta vs z relationship is preserved (unwrapped)
    """
    from double_helix.z_mapping import calibrate_double_helix_psf
    
    psf_stack = ImageStack(psf_stack_arr, md, haveGUI=False)
    cal = calibrate_double_helix_psf(
        psf_stack,
        'double_helix.DoubleGaussFit',
        roi_half_size=md['Analysis.ROISize'],
        filter_sigma=md['Analysis.DetectionFilterSigma'],
        lobe_sep_guess=md['Analysis.LobeSepGuess'],
        lobe_sigma_guess=md['Analysis.SigmaGuess']
    )
    c_ind = 0  # test first channel

    # Unwrap theta to handle the 2π periodicity properly
    unwrapped_theta_gt = np.unwrap(_theta, np.pi/2, period=np.pi)
    unwrapped_theta_cal = np.unwrap(np.asarray(cal[c_ind]['theta']), np.pi/2, period=np.pi)

    # Check that Z positions in calibration match expected Z positions
    np.testing.assert_almost_equal(_z, cal[c_ind]['z'], decimal=1)
    
    # Check that theta vs z relationship in calibration matches input relationship
    np.testing.assert_almost_equal(unwrapped_theta_cal, unwrapped_theta_gt, decimal=1)


# ============================================================================
# TEST 2: Z Mapping with Focus Step → Recovered Sample Positions
# ============================================================================
def test_z_focus_mapping():
    """
    Test that Z mapping correctly recovers sample positions despite a focus step.
    
    This test simulates the scenario:
    1. Sample contains multiple Z positions (of course)
    2. Objective focuses to a different z (focus step)
    3. Same sample structure is imaged again 

    The mapping should recover that the sample z positions are the same before
    and after the focus step, as if the step had never occurred.
    """
    from double_helix.z_mapping import calibrate_double_helix_psf
    from double_helix.recipes.DH_mappings import DoubleHelixMapZ
    import json
    import tempfile
    import os

    # Generate calibration
    psf_stack = ImageStack(psf_stack_arr, md, haveGUI=False)
    cal = calibrate_double_helix_psf(
        psf_stack,
        'double_helix.DoubleGaussFit',
        roi_half_size=md['Analysis.ROISize'],
        filter_sigma=md['Analysis.DetectionFilterSigma'],
        lobe_sep_guess=md['Analysis.LobeSepGuess'],
        lobe_sigma_guess=md['Analysis.SigmaGuess']
    )
    c_ind = 0  # test first channel

    # Add z_range to calibration (required by DoubleHelixMapZ)
    for chan_cal in cal:
        chan_cal['z_range'] = (_z[0] - 1, _z[-1] + 1)

    # Save calibration to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dh_json', delete=False, encoding='utf8') as f:
        temp_path = f.name
        json.dump(cal, f)
    
    try:
        # Map z positions using the double helix mapping
        # Input is objective position (focus), output is sample position (dh_z) plus input
        recipe_results = DoubleHelixMapZ(
            calibration_location=temp_path,
            target_knot_spacing=2 * step_size + 1
        ).apply(input_name=gt_focus)
        
        mapped = recipe_results['dh_localizations']
        
        # The mapping should recover that the sample is at the same z positions before
        # and after the focus step (as if the focus step had not happened)
        expected_z_mapped = np.hstack([trimmed_z, trimmed_z])
        np.testing.assert_allclose(mapped['z'], expected_z_mapped, atol=1.1)
        
    finally:
        os.unlink(temp_path)
