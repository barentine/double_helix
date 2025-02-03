This plugin enables double-helix point-spread function detection and localization fitting within the [Python Microscopy Environment (PYME)](https://python-microscopy.org/)

## Key Features
- highly efficient detection of double-helix point-spread functions using just 7 (separable) convolutions
- Double-gaussian localization fitting parameterized on lobe separation and orientation, which enables direct estimates of their uncertainties

## Installation / Set up
1. [Install PYME](https://python-microscopy.org/doc/Installation/Installation.html), using your perfered method. The simplest approach is to use an [executable installer](https://python-microscopy.org/downloads/), though conda or a development installation provided more flexibility for fast upgrades or developing things yourself. As of 2025/01, C. Soeller's [test environment setup tools](https://github.com/csoeller/PYME-test-env) make a great starting point for deveopment installations.
2. Using git, clone this repository to your machine.
3. In a terminal/shell with your conda environment activated, run `python setup.py develop` from the top `double_helix` directory to install this plugin. If you used an executable installer for PYME, look for a "Anaconda prompt (python-microscopy)" shell in your start menu.
4. Check that PYME now finds this pluging by e.g. opening PYMEImage, potentially in test mode `PYMEImage -t` and clicking the `Modules` menu and looking for `DH_Calibration` up near the very top (you may have to scroll up).
5. It is imperative to have correct metadata for good localization. If your data was not acquired using PYME, please see PYME's [analyzing foreign data documentation](https://python-microscopy.org/doc/AnalysingForeignData.html).

## Usage

### PSF calibration
Calibrating your point-spread function is a great way to start. While this calibration is not necessary for fitting, it is used to eventually convert the orientation of the double-helix to a z position for a localized molecule.
Additionally, calibrating your PSF will help determine optimal detection settings.

1. Open your bead stack in PYMEImage. If you already have a cropped Z-stack of a single bead, your "extracted PSF", proceed to the next step. If you have multiple beads in the same field of view, you may wish to enable the PYME `psfExtraction` module. The extraction process is described in the [PYME documentation](https://python-microscopy.org/doc/PSFExtraction.html).
2. Once you have your extracted PSF open in PYMEImage, load the double-helix calibration module by clicking on `DH_Calibration` in the `Modules` menu near the top.
4. Save the resulting calibration as a `.dh_json` file. The calibration plots will be automatically saved in the same directory.


### Localization fitting
Start by familiarizing yourself with the localization workflow in PYME, documented [here](https://python-microscopy.org/doc/LocalisationAnalysis.html#).
PYME can facilitate distributed batch analysis of many series, but we will describe how to localize a single series here.

1. Start the `PYMEClusterOfOne`.
2. Open your image series in `PYMEImage`
3. If your image is not an `.h5` file type, you will likely need to activate the `LMAnalysis` module
4. Select your prefered background subtraction. We recommend checking `Use percentile for background`.
5. If you have variance, dark, or flat maps to load, do so. Otherwise, scalar values will be used as listed in the metadata.
6. Enter your double helix deteciton settings, as informed by your PSF calibration. The expected lobe separation [nm], and expected lobe sigma [nm] are starting points for each fit, and should be set using reasonable ~middle values from your PSF calibration.
   Enter a value for the detection filter sigma as optimized above.
7. Click `Test This Frame` and see how well molecules are detected and fitted. Adjust the `Thresh` value, and any other settings as necessary.

### Z-lookup for fitted localizations

1. Open a set of localized points in `PYMEVis`.
2. In the `Corrections` menu, click `Double Helix > Map and Filter on DH parameters.
3. Select the dh_json calibration from above, and if necessary, adjust the target knot spacing. A cubic spline is fitted to the calibration data and used to look-up individual localization's z position. The target knot spacing of that spline is adjustable, within some limits of the spacing of the original calibration - increase the knot spacing if the default setting is too small for your calibration data.
4. The `z` column of the output datasource will be automatically adjusted to now include the `dh_z` position (added to any `focus` information of individual localizations, potentially with a foreshortening correciotn). Several double-helix specific filters will be automatically added to the pipeline and may need to be adjusted for your use case.
