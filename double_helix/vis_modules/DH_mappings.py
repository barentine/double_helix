
import logging

logger=logging.getLogger(__name__)


class DHMapper(object):
    def __init__(self, vis_frame):
        self.vis_frame = vis_frame
        self.pipeline = vis_frame.pipeline

        logging.debug('Adding menu items for double helix localizations')

        vis_frame.AddMenuItem('Corrections>Double Helix', 'Map and filter on DH parameters',
                              self.OnMapAndFilter)

    def OnMapAndFilter(self, wx_event):
        from double_helix.recipes.DH_mappings import DoubleHelixMapZ
        from PYME.recipes.tablefilters import FilterTable
        import numpy as np

        recipe = self.pipeline.recipe

        dh_mapper = DoubleHelixMapZ(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                    output_name='dh_mapped')
        
        try:
            dh_mapper.configure_traits(kind='modal')
        except:
            import wx
            wx.SizerFlags.DisableConsistencyChecks()
            dh_mapper.configure_traits(kind='modal')
        
        recipe.add_modules_and_execute([dh_mapper,])
        self.pipeline.selectDataSource('dh_mapped')
        
        # ---------- set up some default filter ranges to be fairly permissive:
        lobe_sep = self.pipeline.mdh['Analysis.LobeSepGuess']
        lobe_sep_half_range = 0.5 * lobe_sep
        lobe_sep_min = lobe_sep - lobe_sep_half_range
        lobe_sep_max = lobe_sep + lobe_sep_half_range

        sigma = self.pipeline.mdh['Analysis.SigmaGuess']
        sigma_half_range = 0.5 * sigma
        sigma_min = sigma - sigma_half_range
        sigma_max = sigma + sigma_half_range
        
        xy_detection_residual = self.pipeline.mdh['Analysis.ROISize'] * 0.25 * (self.pipeline.mdh.voxelsize_nm.x + 
                                       self.pipeline.mdh.voxelsize_nm.y)  # [nm]

        # add the filtering recipe module
        dh_filter = FilterTable(recipe, inputName=self.pipeline.selectedDataSourceKey,
                                filters={
                                    'error_x' : [0, 50],  # [nm]
                                    'error_y': [0, 50],  # [nm]
                                    'dh_z_error' : [0, 100], # [nm]
                                    'sig' : [sigma_min, sigma_max], #[nm]
                                    'fitResults_lobesep': [lobe_sep_min, lobe_sep_max],  # [nm]
                                    'fitError_theta': [0, 0.3],  # [rad]
                                    'fitResults_A0' : [0, np.finfo(np.float32).max], # ADC Counts
                                    'fitResults_A1' : [0, np.finfo(np.float32).max], # ADC Counts
                                    'dh_amp_ratio' : [0, 1],
                                    'dh_xy_detection_residual': [0, xy_detection_residual],
                                    'dh_sigma_residual': [-sigma_half_range, sigma_half_range],
                                    'dh_lobesep_residual': [-lobe_sep_half_range, lobe_sep_half_range]
                                    # 'n_photoelectrons': [100, 10000],  # [pe-]
                                    }, outputName='dh_filtered')
        
        
        recipe.add_modules_and_execute([dh_filter,])
        self.pipeline.selectDataSource('dh_filtered')

        self.vis_frame.RefreshView()


def Plug(vis_frame):  # plug this into PYMEVis
    DHMapper(vis_frame)
