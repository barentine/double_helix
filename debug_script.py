
# example debug script for working without a GUI on e.g. the Z mapping

from PYME.recipes import recipe, modules

yaml = r"""- localisations.AddPipelineDerivedVars:
    inputEvents: ''
    inputFitResults: _FitResults
    outputLocalizations: Localizations
- localisations.ProcessColour:
    input: Localizations
    output: colour_mapped
- tablefilters.FilterTable:
    filters:
      error_x:
      - 0
      - 30
      error_y:
      - 0
      - 30
      sig:
      - 150
      - 275
    inputName: colour_mapped
    outputName: filtered_localizations
- double_helix.DoubleHelixMapZ:
    calibration_location: C:\Users\aesb\PYMEData\Users\aesb\Downloads\analysis\red_channel.dh_json
    input_name: filtered_localizations
    output_name: dh_mapped
"""
rec = recipe.Recipe.fromYAML(yaml)

filename = 'C:\\Users\\aesb\\PYMEData\\Users\\aesb\\Downloads\\analysis\\FOV2_ER_500mW_647_p02exp_Gain300_X2.h5r'

rec.loadInput(filename, '')

rec.execute()