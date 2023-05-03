
import logging

logger=logging.getLogger(__name__)


class DHMapper(object):
    def __init__(self, vis_frame):
        self.vis_frame = vis_frame
        self.pipeline = vis_frame.pipeline

        logging.debug('Adding menu items for double helix localizations')

        vis_frame.AddMenuItem('Corrections>Double Helix>Map and filter on DH parameters',
                              self.OnMapAndFilter)

    def OnMapAndFilter(self, wx_event):
        from double_helix.recipes.DH_mappings import DHMappings
        from PYME.recipes.tablefilters import FilterTable

        recipe = self.pipeline.recipe

        dh_mapper = DHMappings(recipe, input_name=self.pipeline.selectedDataSourceKey,
                               output_name='dh_mapped')
        
        # FIXME - add a configure modal thing here once we have z-mapping set up for DHMappings
        recipe.add_modules_and_execute([dh_mapper,])
         self.pipeline.selectDataSource('dh_mapped')
        
        lobe_sep = np.median(pipeline)
        lobe_sep_half_range = 0.15 * lobe_sep
        lobe_sep_min = lobe_sep - lobe_sep_half_range
        lobe_sep_max = lobe_sep + lobe_sep_half_range

        dh_filter = FilterTable(recipe, inputName=self.pipeline.selectedDataSourceKey,
                                filters={
                                    'error_x' : [0, 30],  # [nm]
                                    'error_y': [0, 30],  # [nm]
                                    'lobe_separation': [lobe_sep_min, lobe_sep_max],  # [nm]
                                    'error_theta': [0, 0.3],  # [rad]
                                    'n_photoelectrons': [100, 10000],  # [pe-]
                                    }, outputName='dh_filtered')
        
        
        recipe.add_modules_and_execute([dh_filter,])
        self.pipeline.selectDataSource('dh_filtered')

        self.vis_frame.RefreshView()


def Plug(vis_frame):  # plug this into PYMEVis
    DHMapper(vis_frame)
