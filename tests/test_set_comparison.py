import numpy as np
import pytest
from PYME.IO.tabular import DictSource


def make_source(x, y, z):
    return DictSource({
        'x': np.asarray(x, dtype=float),
        'y': np.asarray(y, dtype=float),
        'z': np.asarray(z, dtype=float),
    })


def test_align_xy_corrects_offset():
    """AlignXY shifts x/y by the median matched offset relative to the reference."""
    from double_helix.recipes.set_comparison import AlignXY

    x_offset = 10.0
    y_offset = -5.0

    input_locs = make_source(
        x=[100.0, 200.0, 300.0],
        y=[40.0, 50.0, 60.0],
        z=[0.0, 0.0, 0.0],
    )
    reference_locs = make_source(
        x=[100.0 - x_offset, 200.0 - x_offset, 300.0 - x_offset],
        y=[40.0 - y_offset, 50.0 - y_offset, 60.0 - y_offset],
        z=[0.0, 0.0, 0.0],
    )

    module = AlignXY()
    result = module.apply(
        input_name=input_locs,
        input_reference_localizations=reference_locs,
    )

    aligned = result['aligned_localizations']

    expected_x = np.array([100.0, 200.0, 300.0]) - x_offset
    expected_y = np.array([40.0, 50.0, 60.0]) - y_offset

    np.testing.assert_allclose(aligned['x'], expected_x)
    np.testing.assert_allclose(aligned['y'], expected_y)


def test_align_xy_zero_offset():
    """AlignXY leaves coordinates unchanged when input and reference already match."""
    from double_helix.recipes.set_comparison import AlignXY

    coords = make_source(
        x=[10.0, 20.0, 30.0],
        y=[1.0, 2.0, 3.0],
        z=[0.0, 0.0, 0.0],
    )

    module = AlignXY()
    result = module.apply(
        input_name=coords,
        input_reference_localizations=coords,
    )

    aligned = result['aligned_localizations']
    np.testing.assert_allclose(aligned['x'], coords['x'])
    np.testing.assert_allclose(aligned['y'], coords['y'])


# ---------------------------------------------------------------------------
# JaccardAndRMSE tests
# ---------------------------------------------------------------------------

def test_jaccard_and_rmse_perfect_match():
    """Perfect overlap: Jaccard=100, recall=1, precision=1, RMSE=0."""
    from double_helix.recipes.set_comparison import JaccardAndRMSE

    locs = make_source(
        x=[0.0, 100.0, 200.0],
        y=[0.0, 100.0, 200.0],
        z=[0.0,   0.0,   0.0],
    )

    module = JaccardAndRMSE()
    result = module.apply(
        input_name=locs,
        input_reference_localizations=locs,
    )

    m = result['jaccard_and_rmse']
    np.testing.assert_allclose(m['jaccard_index'], [100.0])
    np.testing.assert_allclose(m['recall'],        [1.0])
    np.testing.assert_allclose(m['precision'],     [1.0])
    np.testing.assert_allclose(m['lateral_rmse'],  [0.0])
    np.testing.assert_allclose(m['axial_rmse'],    [0.0])


def test_jaccard_and_rmse_error_annotated_columns():
    """Perfect match: per-localization lateral and axial errors are all zero."""
    from double_helix.recipes.set_comparison import JaccardAndRMSE

    locs = make_source(
        x=[0.0, 100.0, 200.0],
        y=[0.0, 100.0, 200.0],
        z=[0.0,   0.0,   0.0],
    )

    module = JaccardAndRMSE()
    result = module.apply(
        input_name=locs,
        input_reference_localizations=locs,
    )

    annotated = result['error_annonated']
    assert 'lateral_error' in annotated.keys()
    assert 'axial_error'   in annotated.keys()
    assert len(annotated['lateral_error']) == len(locs['x'])
    np.testing.assert_allclose(annotated['lateral_error'], 0.0)
    np.testing.assert_allclose(annotated['axial_error'],   0.0)


def test_jaccard_and_rmse_no_overlap():
    """Results far outside the linking radius: Jaccard=0, errors filled with link_radius_nm."""
    from double_helix.recipes.set_comparison import JaccardAndRMSE

    # results displaced 1000 nm from reference in x — well beyond default 250 nm radius
    reference = make_source(x=[0.0, 100.0], y=[0.0, 100.0], z=[0.0, 0.0])
    results   = make_source(x=[0.0 + 1000.0, 100.0 + 1000.0], y=[0.0, 100.0], z=[0.0, 0.0])

    module = JaccardAndRMSE()
    result = module.apply(
        input_name=results,
        input_reference_localizations=reference,
    )

    m = result['jaccard_and_rmse']
    np.testing.assert_allclose(m['jaccard_index'], [0.0])
    np.testing.assert_allclose(m['recall'],        [0.0])
    np.testing.assert_allclose(m['precision'],     [0.0])

    # unmatched localizations should be filled with the link radius sentinel value
    annotated = result['error_annonated']
    np.testing.assert_allclose(annotated['lateral_error'], module.link_radius_nm)
    np.testing.assert_allclose(annotated['axial_error'],   module.link_radius_nm)


def test_jaccard_and_rmse_known_per_localization_errors():
    """Per-localization lateral and axial errors match known geometry."""
    from double_helix.recipes.set_comparison import JaccardAndRMSE

    # Two results each offset from a single reference at the origin:
    #   result 0: (dx=3, dy=0, dz=10) -> lateral=3, axial=10
    #   result 1: (dx=0, dy=4, dz= 5) -> lateral=4, axial= 5
    # Both are within the default 250 nm linking radius.
    reference = make_source(x=[0.0, 0.0], y=[0.0, 0.0], z=[0.0, 0.0])
    results   = make_source(x=[3.0, 0.0], y=[0.0, 4.0], z=[10.0, 5.0])

    module = JaccardAndRMSE()
    result = module.apply(
        input_name=results,
        input_reference_localizations=reference,
    )

    annotated = result['error_annonated']
    np.testing.assert_allclose(annotated['lateral_error'], [3.0, 4.0])
    np.testing.assert_allclose(annotated['axial_error'],   [10.0, 5.0])

