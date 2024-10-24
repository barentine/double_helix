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
