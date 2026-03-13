
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


def test_DoubleGaussFit_FindAndFit_XYTheta():
    """
    Make sure that DoubleGaussFit.DumbellFitFactory.FindAndFit can find the correct X, Y, Theta position of Double Helices in a test stack with
    multiple Z positions, and a lateral offset which would cause an error for an incorrectly transposed stack.
    """
    # Generate a test image stack with an X offset to check for correct handling of XY conventions
    # especially combined with theta definitions.
    x_shift = 1.25 * md.voxelsize_nm.x  # 1.25 pixel shift in X
    test_stack_arr = np.empty(
        (2 * md['Analysis.ROISize'] + 1, 2 * md['Analysis.ROISize'] + 1, len(trimmed_theta)),
        float
    )
    for zind in range(len(trimmed_theta)):
        test_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
            [A0, A0, x_shift, 0, trimmed_theta[zind], md['Analysis.LobeSepGuess'],
            md['Analysis.SigmaGuess'], 20],
            md,
            x=md['voxelsize.x'] * md['Analysis.ROISize'],  # Add X offset
            y=md['voxelsize.y'] * md['Analysis.ROISize'],
            roiHalfSize=md['Analysis.ROISize']
        )
    test_stack = ImageStack(test_stack_arr, md)

    # Create metadata with tIndex for results packing
    # tIndex must be set as a direct attribute - DictMDHandler does not expose
    # dict keys as attributes, so metadata.tIndex would otherwise return None.
    test_md = MetaDataHandler.DictMDHandler(md)
    test_md.tIndex = 0

    # Run FindAndFit on each z-frame and collect fit results
    roi_size = md['Analysis.ROISize']
    noise_sigma = np.ones((2 * roi_size + 1, 2 * roi_size + 1, 1))

    x0_results = []
    y0_results = []
    theta_results = []

    for zind in range(len(trimmed_theta)):
        frame_data = test_stack_arr[:, :, zind:zind + 1]  # (x, y, 1) in PYME convention
        factory = DoubleGaussFit.DumbellFitFactory(frame_data, test_md, noiseSigma=noise_sigma)
        results = factory.FindAndFit(threshold=1.0, cameraMaps=None)

        assert len(results) == 1, (
            f"Expected exactly 1 detection at z-index {zind}, found {len(results)}"
        )

        x0_results.append(float(results['fitResults']['x0'][0]))
        y0_results.append(float(results['fitResults']['y0'][0]))
        theta_results.append(float(results['fitResults']['theta'][0]))

    x0_arr = np.array(x0_results)
    y0_arr = np.array(y0_results)
    theta_arr = np.array(theta_results)

    # Check that the x_shift appears in x0, not y0.
    # The evalModel grid has the same origin for x and y, so x0 - y0 should equal x_shift.
    # A transposed data array would produce x0 - y0 ≈ -x_shift instead, causing this to fail.
    np.testing.assert_allclose(
        x0_arr - y0_arr,
        x_shift * np.ones(len(trimmed_theta)),
        atol=0.1 * md.voxelsize_nm.x,  # 0.1 pixel tolerance [nm]
        err_msg="x0 - y0 does not equal x_shift: x/y convention may be transposed"
    )

    # Check that theta is correctly recovered (pi-periodic, since A0 == A1 makes theta
    # and theta+pi produce identical PSFs)
    d = (theta_arr - trimmed_theta) % np.pi  # wrapped to [0, pi)
    theta_err = np.minimum(d, np.pi - d)      # wrapped to [0, pi/2]
    assert np.all(theta_err < 0.1), (
        f"Theta not recovered within 0.1 rad: max error = {theta_err.max():.3f} rad"
    )


def test_DoubleGaussFit_FromPoint_XYTheta():
    """
    Make sure that DoubleGaussFit.DumbellFitFactory.FromPoint can find the correct X, Y, Theta position of a Double Helix in a test stack with
    multiple Z positions, and a lateral offset which would cause an error for an incorrectly transposed stack.
    """
    x_shift = 1.25 * md.voxelsize_nm.x  # 1.25 pixel shift in X
    roi_size = md['Analysis.ROISize']
    test_stack_arr = np.empty(
        (2 * roi_size + 1, 2 * roi_size + 1, len(trimmed_theta)),
        float
    )
    for zind in range(len(trimmed_theta)):
        test_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
            [A0, A0, x_shift, 0, trimmed_theta[zind], md['Analysis.LobeSepGuess'],
             md['Analysis.SigmaGuess'], 20],
            md,
            x=md['voxelsize.x'] * roi_size,
            y=md['voxelsize.y'] * roi_size,
            roiHalfSize=roi_size
        )
    
    test_md = MetaDataHandler.DictMDHandler(md)
    # tIndex must be a direct attribute - DictMDHandler does not expose dict keys as attributes
    test_md.tIndex = 0

    noise_sigma = np.ones((2 * roi_size + 1, 2 * roi_size + 1, 1))

    x0_results = []
    y0_results = []
    theta_results = []

    for zind in range(len(trimmed_theta)):
        frame_data = test_stack_arr[:, :, zind:zind + 1]  # (x, y, 1) in PYME convention
        factory = DoubleGaussFit.DumbellFitFactory(frame_data, test_md, noiseSigma=noise_sigma)
        # Pass the array center as the hint; roiHalfSize=roi_size covers the full frame
        result = factory.FromPoint(x=roi_size, y=roi_size, roiHalfSize=roi_size)

        x0_results.append(float(result['fitResults']['x0']))
        y0_results.append(float(result['fitResults']['y0']))
        theta_results.append(float(result['fitResults']['theta']))

    x0_arr = np.array(x0_results)
    y0_arr = np.array(y0_results)
    theta_arr = np.array(theta_results)

    # Check that the x_shift appears in x0, not y0.
    # A transposed data array would produce x0 - y0 ≈ -x_shift instead.
    np.testing.assert_allclose(
        x0_arr - y0_arr,
        x_shift * np.ones(len(trimmed_theta)),
        atol=0.1 * md.voxelsize_nm.x,  # 0.1 pixel tolerance [nm]
        err_msg="x0 - y0 does not equal x_shift: x/y convention may be transposed"
    )

    # Check that theta is correctly recovered (pi-periodic, since A0 == A1 makes theta
    # and theta+pi produce identical PSFs)
    d = (theta_arr - trimmed_theta) % np.pi  # wrapped to [0, pi)
    theta_err = np.minimum(d, np.pi - d)      # wrapped to [0, pi/2]
    assert np.all(theta_err < 0.1), (
        f"Theta not recovered within 0.1 rad: max error = {theta_err.max():.3f} rad"
    )


def test_DoubleHelixFindAndFit_XYTheta():
    """
    Make sure that double_helix.recipes.DH_mappings.DetectDoubleHelices has the correct X, Y, and Theta conventions, by looking
    at a test stack with multiple Z positions and a lateral offset which would cause an error for an incorrectly transposed stack.
    """
    from double_helix.recipes.DH_mappings import DetectDoubleHelices
    from unittest.mock import patch
    import numpy as np

    x_shift = 1.25 * md.voxelsize_nm.x  # 1.25 pixel shift in X
    roi_size = md['Analysis.ROISize']
    n_frames = len(trimmed_theta)

    # Build z-stack with one DH per frame, each with the same x_shift but varying theta
    test_stack_arr = np.empty(
        (2 * roi_size + 1, 2 * roi_size + 1, n_frames),
        float
    )
    for zind in range(n_frames):
        test_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
            [A0, A0, x_shift, 0, trimmed_theta[zind], md['Analysis.LobeSepGuess'],
             md['Analysis.SigmaGuess'], 20],
            md,
            x=md['voxelsize.x'] * roi_size,
            y=md['voxelsize.y'] * roi_size,
            roiHalfSize=roi_size
        )

    test_stack = ImageStack(test_stack_arr, md, haveGUI=False)

    # Patch fitTask.calcSigma to return a flat noise sigma of 1.0 per pixel,
    # avoiding dependency on camera map files in the test environment.
    flat_sigma = np.ones((2 * roi_size + 1, 2 * roi_size + 1))
    with patch('PYME.localization.remFitBuf.fitTask.calcSigma', return_value=flat_sigma):
        outputs = DetectDoubleHelices(
            lobe_sep_nm=md['Analysis.LobeSepGuess'],
            lobe_sigma_nm=md['Analysis.SigmaGuess'],
            fit_roi_half_size=roi_size,
            thresh=1.0,
        ).apply(input_image=test_stack)

    detections = outputs['dh_detections']

    assert len(detections['x']) == n_frames, (
        f"Expected {n_frames} detections (one per frame), got {len(detections['x'])}"
    )

    x_arr = np.array(detections['x'])
    y_arr = np.array(detections['y'])
    angle_arr = np.array(detections['angle'])

    # Check that the x_shift appears in x, not y.
    # Detection is at integer pixels, so tolerance is 1 pixel.
    # A transposed frame would produce x - y ≈ -x_shift instead, failing this assertion.
    np.testing.assert_allclose(
        x_arr - y_arr,
        x_shift * np.ones(n_frames),
        atol=md.voxelsize_nm.x,  # 1 pixel tolerance for integer detection
        err_msg="x - y does not equal x_shift: x/y convention may be transposed"
    )

    # Check that angle is correctly recovered (pi-periodic)
    d = (angle_arr - trimmed_theta) % np.pi  # wrapped to [0, pi)
    angle_err = np.minimum(d, np.pi - d)      # wrapped to [0, pi/2]
    assert np.all(angle_err < 0.1), (
        f"angle not recovered within 0.1 rad: max error = {angle_err.max():.3f} rad"
    )


def test_wobble_correction():
    """
    Test that if the Z calibration stack has Z-dependent X and Y offsets ("wobble") they
    are correctly calibrated and can be subtracted from the localizations during mapping.

    This test:
    1. Builds a calibration PSF stack where the PSF center shifts in x and y with z
    2. Checks that calibrate_double_helix_psf recovers the known wobble pattern.
    3. Creates localizations whose raw x, y contain the same z-dependent wobble.
    4. Verifies that DoubleHelixMapZ with correct_wobble=True removes the wobble.
    5. Verifies that with correct_wobble=False the z-dependent variation is retained.
    """
    from double_helix.z_mapping import calibrate_double_helix_psf
    from double_helix.recipes.DH_mappings import DoubleHelixMapZ
    import json
    import tempfile
    import os

    roi_size = md['Analysis.ROISize']
    n_cal = len(_z)

    # Define a sinusoidal wobble pattern that is zero at the central calibration slice
    wobble_amplitude = 150  # nm (~1.25 pixels), large enough to dominate fitting noise
    t = np.linspace(-np.pi, np.pi, n_cal)
    x0_shifts = wobble_amplitude * np.sin(t)        # zero at center (t=0)
    y0_shifts = wobble_amplitude * np.sin(2 * t) / 2  # independent shape for y

    # Build a calibration PSF stack with the known wobble baked in as x0, y0 shifts
    wobble_psf_arr = np.empty(
        (2 * roi_size + 1, 2 * roi_size + 1, n_cal),
        float
    )
    for zind in range(n_cal):
        wobble_psf_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
            [A0, A0, x0_shifts[zind], y0_shifts[zind], _theta[zind],
             md['Analysis.LobeSepGuess'], md['Analysis.SigmaGuess'], 20],
            md,
            x=md['voxelsize.x'] * roi_size,
            y=md['voxelsize.y'] * roi_size,
            roiHalfSize=roi_size
        )

    wobble_psf_stack = ImageStack(wobble_psf_arr, md, haveGUI=False)
    cal = calibrate_double_helix_psf(
        wobble_psf_stack,
        'double_helix.DoubleGaussFit',
        roi_half_size=roi_size,
        lobe_sep_guess=md['Analysis.LobeSepGuess'],
        lobe_sigma_guess=md['Analysis.SigmaGuess']
    )
    c_ind = 0

    # Expected wobble in calibration = shifts relative to the central slice
    x_central_idx = n_cal // 2
    x_wobble_expected = x0_shifts - x0_shifts[x_central_idx]
    y_wobble_expected = y0_shifts - y0_shifts[x_central_idx]

    # Check that the calibration correctly extracted the wobble pattern
    np.testing.assert_allclose(
        cal[c_ind]['x_wobble'], x_wobble_expected, atol=10.0,
        err_msg="Calibrated x_wobble does not match known wobble pattern"
    )
    np.testing.assert_allclose(
        cal[c_ind]['y_wobble'], y_wobble_expected, atol=10.0,
        err_msg="Calibrated y_wobble does not match known wobble pattern"
    )

    # Add z_range required by DoubleHelixMapZ
    for chan_cal in cal:
        chan_cal['z_range'] = (_z[0] - 1, _z[-1] + 1)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dh_json', delete=False, encoding='utf8') as f:
        temp_path = f.name
        json.dump(cal, f)

    try:
        # Synthetic localizations: true position is zero; raw x, y include the z-dependent wobble.
        # Use trimmed z range to avoid spline edge effects.
        true_x = 0.0
        true_y = 0.0
        x_wobble_trimmed = x_wobble_expected[step_size_ind:-step_size_ind]
        y_wobble_trimmed = y_wobble_expected[step_size_ind:-step_size_ind]

        loc_dict = {
            'z': np.zeros_like(trimmed_z),       # objective at z=0; dh_z will equal trimmed_z
            'x': true_x + x_wobble_trimmed,
            'y': true_y + y_wobble_trimmed,
            'fitResults_theta': trimmed_theta,
            'fitError_theta': 0.001 * np.ones_like(trimmed_theta),
            'fitResults_sigma': md['Analysis.SigmaGuess'] * np.ones_like(trimmed_theta),
            'fitResults_lobesep': md['Analysis.LobeSepGuess'] * np.ones_like(trimmed_theta),
            'fitResults_A0': A0 * np.ones_like(trimmed_theta),
            'fitResults_A1': A0 * np.ones_like(trimmed_theta),
            'startParams_x0': true_x * np.ones_like(trimmed_theta),
            'startParams_y0': true_y * np.ones_like(trimmed_theta),
            'fitError_sigma': np.ones_like(trimmed_theta),
            'fitError_lobesep': np.ones_like(trimmed_theta),
        }
        locs = DictSource(loc_dict)
        locs.mdh = MetaDataHandler.DictMDHandler(md)

        # --- Wobble correction ON ---
        result_corrected = DoubleHelixMapZ(
            calibration_location=temp_path,
            target_knot_spacing=2 * step_size + 1,
            correct_wobble=True
        ).apply(input_name=locs)
        corrected = result_corrected['dh_localizations']

        # After correction x and y should be approximately constant (wobble removed)
        np.testing.assert_allclose(
            corrected['x'], true_x * np.ones_like(trimmed_z), atol=1.0,
            err_msg="Wobble correction did not remove x wobble"
        )
        np.testing.assert_allclose(
            corrected['y'], true_y * np.ones_like(trimmed_z), atol=1.0,
            err_msg="Wobble correction did not remove y wobble"
        )

        # --- Wobble correction OFF ---
        result_uncorrected = DoubleHelixMapZ(
            calibration_location=temp_path,
            target_knot_spacing=2 * step_size + 1,
            correct_wobble=False
        ).apply(input_name=locs)
        uncorrected = result_uncorrected['dh_localizations']

        # Without correction, x and y should retain z-dependent variation
        assert np.std(uncorrected['x']) > 0.1 * wobble_amplitude, (
            f"Without wobble correction, x should vary with z; std = {np.std(uncorrected['x']):.1f} nm"
        )
        assert np.std(uncorrected['y']) > 0.1 * wobble_amplitude, (
            f"Without wobble correction, y should vary with z; std = {np.std(uncorrected['y']):.1f} nm"
        )

    finally:
        os.unlink(temp_path)


def test_startParams_fitResults_theta_bias():
    """
    Check that there is no systematic theta bias between startParams_theta
    (initial estimate from Detector.extract_candidates via the steerable filter) and
    fitResults_theta (final Gaussian-fit value) in FindAndFit.

    A non-zero mean bias would indicate that the detector and fitter use different
    theta conventions or that the steerable filter introduces a systematic offset.

    Because A0 == A1 the PSF is pi-periodic in theta, so differences are wrapped to
    [-pi/2, pi/2] before computing the mean bias.
    """
    roi_size = md['Analysis.ROISize']
    n_frames = len(trimmed_theta)

    # Build a z-stack: one DH per frame with known theta, centered, no lateral offset
    test_stack_arr = np.empty(
        (2 * roi_size + 1, 2 * roi_size + 1, n_frames),
        float
    )
    for zind in range(n_frames):
        test_stack_arr[:, :, zind], _, _, _ = DoubleGaussFit.DumbellFitFactory.evalModel(
            [A0, A0, 0, 0, trimmed_theta[zind], md['Analysis.LobeSepGuess'],
             md['Analysis.SigmaGuess'], 20],
            md,
            x=md['voxelsize.x'] * roi_size,
            y=md['voxelsize.y'] * roi_size,
            roiHalfSize=roi_size
        )

    test_md = MetaDataHandler.DictMDHandler(md)
    test_md.tIndex = 0

    noise_sigma = np.ones((2 * roi_size + 1, 2 * roi_size + 1, 1))

    start_theta = []
    fit_theta = []

    for zind in range(n_frames):
        frame_data = test_stack_arr[:, :, zind:zind + 1]
        factory = DoubleGaussFit.DumbellFitFactory(frame_data, test_md, noiseSigma=noise_sigma)
        results = factory.FindAndFit(threshold=1.0, cameraMaps=None)

        assert len(results) == 1, (
            f"Expected exactly 1 detection at z-index {zind}, found {len(results)}"
        )

        start_theta.append(float(results['startParams']['theta'][0]))
        fit_theta.append(float(results['fitResults']['theta'][0]))

    start_theta = np.array(start_theta)
    fit_theta = np.array(fit_theta)

    # Wrap the per-frame difference to [-pi/2, pi/2] to handle the pi-periodicity of the PSF
    delta = (fit_theta - start_theta + np.pi / 2) % np.pi - np.pi / 2
    mean_bias = np.mean(delta)

    assert abs(mean_bias) < 0.01, (
        f"Systematic theta bias between startParams and fitResults: "
        f"mean bias = {mean_bias:.4f} rad"
    )
