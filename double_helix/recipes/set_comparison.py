from PYME.recipes.base import ModuleBase, register_module
from PYME.recipes.traits import Input, Output, FileOrURI, Float, Enum, Int, List, Bool
from PYME.IO import tabular
from PYME.IO import MetaDataHandler
from matplotlib import image
import numpy as np
import logging
logger = logging.getLogger(__name__)

@register_module('AlignXY')
class AlignXY(ModuleBase):
    """
    Shift a set of localizations by the offset in x and y to approximately align to another,
    e.g. a ground truth dataset
    """
    input_name = Input('localizations')
    input_reference_localizations = Input('reference_localizations')
    xy_link_radius_nm = Float(250, desc='linking radius to match localizations to reference localizations, in nm')
    output_name = Output('aligned_localizations')

    def run(self, input_name, input_reference_localizations):
        from scipy.spatial import cKDTree

        # link nearest neighbor within spec'd radius to match localizations to reference localizations, then
        # calculate the median offset:
        gt_coords = np.vstack((input_reference_localizations['x'], input_reference_localizations['y'])).T
        res_coords = np.vstack((input_name['x'], input_name['y'])).T
        gt_tree = cKDTree(gt_coords)
        nn_distances, nn_indices = gt_tree.query(res_coords)
        matched = nn_distances <= self.xy_link_radius_nm
        x_offset = np.median(input_name['x'][matched] - input_reference_localizations['x'][nn_indices[matched]])
        y_offset = np.median(input_name['y'][matched] - input_reference_localizations['y'][nn_indices[matched]])

        

        # calculate mean offset in x and y between input localizations and reference localizations
        # mean_x_offset = np.mean(input_name['x']) - np.mean(input_reference_localizations['x'])
        # mean_y_offset = np.mean(input_name['y']) - np.mean(input_reference_localizations['y'])

        logging.info(f"Calculated x offset: {x_offset:.3f} nm")
        logging.info(f"Calculated y offset: {y_offset:.3f} nm")

        # apply mean offset to input localizations
        aligned_loc = tabular.MappingFilter(input_name,
            x_offset=x_offset,
            y_offset=y_offset,
            x='x - x_offset',
            y='y - y_offset'
        )

        try:
            aligned_loc.mdh = MetaDataHandler.NestedClassMDHandler(input_name.mdh)
        except AttributeError:
            pass

        return {
            'output_name': aligned_loc
        }


@register_module('JaccardAndRMSE')
class JaccardAndRMSE(ModuleBase):
    """
    Calculate the Jaccard index, lateral RMSE, and axial RMSE between results and ground truth.

    Lateral, Axial, and 3D error from input localizations to nearest ground truth localization
    will be added to the output.
    """
    input_name = Input('localizations')
    input_reference_localizations = Input('reference_localizations')
    link_radius_nm = Float(250, desc='linking radius to match results localizations to ground truth, in nm')
    output_name = Output('error_annonated')
    output_measurements = Output('jaccard_and_rmse')  # dict with keys 'jaccard_index', 'recall', 'precision', 'lateral_rmse', 'axial_rmse'

    def run(self, input_name, input_reference_localizations):
        from double_helix.fight_club_tools import jaccard_and_rmse
        jaccard_index, recall, precision, lateral_rmse, axial_rmse, matched, nn_indices = jaccard_and_rmse(
            input_name, input_reference_localizations, radius_nm=self.link_radius_nm)

        logger.info(f"Jaccard index: {jaccard_index:.3f}%")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Lateral RMSE: {lateral_rmse:.3f} nm")
        logger.info(f"Axial RMSE: {axial_rmse:.3f} nm")

        # add per-localization error annotations to input localizations table
        # enter linking_radius nm for localizations with no match within linking radius
        lateral_errors = np.full(len(input_name['x']), fill_value=self.link_radius_nm)
        axial_errors = np.full(len(input_name['x']), fill_value=self.link_radius_nm)
        # for matched localizations, calculate lateral and axial errors to nearest ground truth
        # r for lateral residuals
        lateral_errors[matched] = np.sqrt((input_name['x'][matched] - input_reference_localizations['x'][nn_indices[matched]])**2 + \
                                                  (input_name['y'][matched] - input_reference_localizations['y'][nn_indices[matched]])**2)
        # for axial residuals, just calculate absolute difference in z
        axial_errors[matched] = np.abs(input_name['z'][matched] - input_reference_localizations['z'][nn_indices[matched]])

        error_annotated = tabular.MappingFilter(input_name)
        error_annotated.addColumn('lateral_error', lateral_errors)
        error_annotated.addColumn('axial_error', axial_errors)

        try:
            error_annotated.mdh = MetaDataHandler.NestedClassMDHandler(input_name.mdh)
        except AttributeError:
            pass

        measurements = tabular.DictSource({
            'jaccard_index': np.array([jaccard_index]),
            'recall': np.array([recall]),
            'precision': np.array([precision]),
            'lateral_rmse': np.array([lateral_rmse]),
            'axial_rmse': np.array([axial_rmse]),
        })

        return {
            'output_name': error_annotated,
            'output_measurements': measurements,
        }
