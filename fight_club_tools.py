# fight_club utils
# load 3D double helix data from, e.g. https://srm.epfl.ch/srm/dataset/challenge-3D-simulation/beads/index.html
# as tif and save as h5 with metadata.




def load_tif_and_save_as_h5(tif_fn, h5_fn):
    from PYME.IO.image import ImageStack
    from PYME.IO.MetaDataHandler import NestedClassMDHandler
    
    mdh = NestedClassMDHandler()

    # metadata for their simulated datasets:
    # QE: 0.9 e-/photon
    # pixel size: 0.100 um
    mdh['voxelsize.units'] = 'um'
    mdh['voxelsize.x'] = 0.1  # [um]
    mdh['voxelsize.y'] = 0.1
    # bead calibration z step size: 10 nm -> 0.01 um
    mdh['voxelsize.z'] = 0.01  # [um]
    # wavelength: 660 nm
    # NA: 1.49
    # read-out noise: 74.4 [units?] -> sigma, [e-]. See https://www.nature.com/articles/s41592-019-0364-4
    mdh['Camera.ReadNoise'] = 74.4  # [e-]
    # EM gain: 300
    mdh['Camera.TrueEMGain'] = 300
    mdh['Camera.NoiseFactor'] = 1.409
    # spurious noise: poisson, 0.0020 [units?]
    # total gain: QE * EM_gain / e_per_ADU: 6.00
    # electrons per ADU: 45.00 e-/ADU
    mdh['Camera.ElectronsPerCount'] = 45.0  # [e-/ADU]
    # baseline: 100.00 ADU
    mdh['Camera.ADOffset'] = 100.0  # [ADU]
    
    im = ImageStack(filename=tif_fn, mdh=mdh, haveGUI=False)
    im.save(h5_fn)

def jaccard_and_rmse(results, ground_truth, radius_nm=250):
    """
    Calculate the Jaccard index and both lateral and axial RMSE between results 
    and ground truth.

    Parameters
    ----------
    results : table-like with 'x', 'y', 'z' columns
        The localization results to evaluate.
    ground_truth : table-like
        The ground truth localizations to compare against, with 'x', 'y', 'z' columns.
    radius_nm : float, optional
        lateral linking radius to match results localizations to ground truth, in nm. Default is 250 nm.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    gt_coords = np.vstack((ground_truth['x'], ground_truth['y'], ground_truth['z'])).T
    res_coords = np.vstack((results['x'], results['y'], results['z'])).T

    gt_tree = cKDTree(gt_coords)
    # get nearest ground truth point for each result
    dists, nn_indices = gt_tree.query(res_coords)

    matched = dists < radius_nm
    # Γ(S ∩ T)
    n_true_positives = np.sum(matched)
    
    # Γ(S\S ∩ T) --> result localizations that aren't matched
    n_false_positives = np.sum(np.logical_not(matched))# same as: len(results) - n_true_positives
    
    # Γ(T\S ∩ T) --> ground truth localizations that aren't matched
    # NOTE: can't just do len(ground_truth) - n_true_positives because stats are calculated on per
    # localization basis, not necessarily linked molecules.
    unmatched_gt_indices = np.setdiff1d(np.arange(len(ground_truth)), nn_indices[matched])
    n_false_negatives = len(unmatched_gt_indices)
    # n_false_negatives = np.sum(np.bincount(nn_indices[matched], minlength=len(ground_truth)) == 0)
    jaccard_index = 100 * n_true_positives / (n_true_positives + n_false_positives + n_false_negatives)

    recall = n_true_positives / (n_true_positives + n_false_negatives)
    precision = n_true_positives / (n_true_positives + n_false_positives)

    # for RMSE calculations, only consider true positives
    tp_results = res_coords[matched]
    tp_ground_truth = gt_coords[nn_indices[matched]]
    lateral_rmse = np.sqrt( 1 / n_true_positives * np.sum( (tp_results[:, :2] - tp_ground_truth[:, :2])**2 ) )
    axial_rmse = np.sqrt( 1 / n_true_positives * np.sum( (tp_results[:, 2] - tp_ground_truth[:, 2])**2 ) )

    return jaccard_index, recall, precision, lateral_rmse, axial_rmse, matched, nn_indices