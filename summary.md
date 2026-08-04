# NOPP_comparison

Code for comparing NOPP storm-response models (HurryWave, ADCIRC/SWAN, COAWST/WW3) against observations from Hurricane Helene (2024), including NDBC buoys, NOAA/NHERI water-level gauges, and high-water marks.

## Data download / acquisition

- `download_ndbc.ipynb`, `download_ndbc_final.ipynb` - download NDBC bulk stdmet and directional spectral buoy data for a list of stations; write per-buoy and combined CSV/NetCDF outputs.
- `box_search_buoy_data.ipynb` - find active NDBC stations inside a lat/lon box, download stdmet data for a date range, summarize availability to CSV/NetCDF.
- `download_noaa_wl_helene.ipynb` - pull NOAA water-level station data via the CO-OPS API for the Helene event and inventory it.
- `Untitled3.ipynb` - scratch script downloading data for one NDBC station (42099).

## NHERI rapid-deployment gauge processing

- `nheri_rwg_process.ipynb` - build an atmospheric pressure correction file from instrument and met data (example: Cedar Key).
- `nheri_rwg_fix_timing.ipynb` - reconstruct fractional-second timestamps in NHERI wave-gauge CSV files.
- `process_nheri_csv.ipynb` - read raw 16 Hz NHERI pressure-gauge CSVs, clean columns, build precise timestamps, Butterworth-filter and resample.
- `nheri_funcs.py` - helper functions used by the NHERI notebooks.

## Water-level comparisons

- `WL_comparision.ipynb`, `WL_comp_new.ipynb` - compare modeled (Deltares/COAWST) vs. observed water levels at gauge stations; "new" is an updated version of the same workflow.
- `BAMS_FtMeyers_TS.ipynb` - time-series plot of Ft. Myers NOAA water-level data vs. ADCIRC output, styled for a BAMS paper figure.
- `compare_cedar_key_atmos_press.ipynb` - compare NOAA CO-OPS atmospheric pressure at Cedar Key against an NHERI Sentinel tower deployment.
- `hwm.ipynb` - extract high-water-mark data from a photo (pixel-picking via OpenCV) and find peak water level via smoothing/peak detection.

## NDBC buoy / wave comparisons

There are six NDBC buoys in the study area. (A seventh, 42039 south of Pensacola, is considered too far west to include.)  
Four include directional data, but only three of these have spectra.  
Data from the bouys has been downloaded into `\crs\proj\2025_NOPP_comparison\helene_waves` with bulk statistics in .csv files and spectral data in .nc files.   

- `compare_ndbc_hwave_bulk.ipynb` - download and parse NDBC stdmet (bulk wave) data for comparison against model output.
- `compare_ndbc_hurrywave_coawst_bulk.ipynb` - compare NDBC bulk buoy wave stats against HurryWave and COAWST/WW3 output.
- `compare_ndbc_hurrywave_coawst_adcirc_bulk.ipynb` - three-way comparison adding ADCIRC/SWAN bulk wave output.
- `compare_adcirc_spec_bulk.ipynb` - read ADCIRC spectral/bulk output on the unstructured mesh and compare against observations.
- `compare_coawst_spec_bulk.ipynb` - compare COAWST/WW3 spectral output (THREDDS) against bulk observations using Delaunay interpolation.
- `compare_hwave_spec_bulk.ipynb` - contour utilities and comparison of modeled Hwave against spectral/bulk data along extracted shoreline contours.
- `three_panel_hmo.ipynb` - three-panel comparison figure of Hmo from HURRYWAVE, ADCIRC/SWAN, and COAWST/WW3.
- `compare_polar_v0.ipynb` - polar comparison plot, likely directional wave spectra.

## Wave energy flux on the shelf-edge contour

- `find_contour.ipynb` - generate evenly spaced points along smoothed 15-m and 30-m depth contours; save to `contour*.csv`.
- `match_grid_to_contour_v1.ipynb` - interpolate model grid bathymetry/results onto contour points (horizontal + depth-weighted nearest neighbor).
- `extract_efth_on_contour.ipynb` - extract directional wave energy density (efth) at contour points.
- `extract_adcirc_eflux_from_spc.ipynb` - read ADCIRC SWAN `.spc` spectral files and compute wave energy flux at contour points.
- `extract_adcirc_bulk_on_contour.ipynb` - extract bulk wave parameters from the ADCIRC mesh, interpolated to contour points.
- `extract_coawst_eflux_spec.ipynb` - pull COAWST/WW3 spectral output from THREDDS for energy-flux calculation.
- `extract_coawst_eflux_on_contour.ipynb` - compute COAWST wave energy flux at contour points.
- `extract_all3_eflux.ipynb` - combine contour points and HurryWave/ADCIRC/COAWST spectra to compute energy flux from all three models.
- `flux_plots_all3.ipynb` - integrate spectral energy flux over frequency/direction and plot shoreward (onshore) flux comparisons across the three models.
- `process_adcirc.ipynb` - load previously computed bulk flux output at contour points.

## Model loading / inspection

- `load_COAWST_his_file.ipynb` - open a COAWST ROMS history NetCDF file from THREDDS.
- `load_hurrywave.ipynb` - load HurryWave bulk and spectral output; compute Hm0 from spectra by integrating energy density over direction and frequency.
- `load_examine_various.ipynb`, `examine_hurrywave.ipynb` - exploratory loads/inspections of HurryWave history and spectral NetCDF files.
- `read_adcirc.ipynb` - read ADCIRC water level/wave height from a THREDDS station file (`fort.61.nc`-type), find nearest mesh element, interpolate, plot.
- `weighted_Hs_big_box.ipynb` - compute area-weighted Hs/Hs2/wave power over a bounding box using a model-to-model variable-name map; 15-panel Hs plot.
- `weighted_hs_15_panel.ipynb` - geometry/area-weighting helper routines (polygon/box intersection, structured and unstructured grid cell areas) feeding the 15-panel Hs figure.

## Mapping / visualization

- `Observations_map_helene.ipynb` - map the locations of all observation types (buoys, water level gauges, etc.) during Hurricane Helene.
- `map_filtered_peaks.ipynb` - map high-water-mark/filtered peak observations, with overlays for storm track and station type.
- `map_adcirc.ipynb`, `map_coawst.ipynb`, `map_hurrywave.ipynb` - map each model's mesh/grid output with optional Cartopy basemap.
- `kml_from_csv.ipynb` - convert a CSV of points into a KML file for Google Earth.

## Testing / utility

- `test_cg_intermediate.ipynb` - test/derive linear-wave intermediate-depth group velocity (c_g), iterative dispersion-relation solver.
- `test_parse_spc_file.ipynb` - test the SWAN `.spc` spectral file parser (`parse_spc_file.py`).

## Support modules (.py)

- `nheri_funcs.py` - NHERI gauge helper functions.
- `parse_spc_file.py` - SWAN `.spc` spectral file reader.
- `plot_contour15.py` - contour plotting helper.
- `scat_stats.py` - scatter/statistics routines for modeled vs. observed comparisons.
- `spec_plot_funcs.py` - spectral plotting functions.
- `storm_coords.py` - storm-relative coordinate transforms and spatial helpers.
- `wave_stats.py` - wave spectra statistics.
- `wl_stats.py` - water-level statistics.

## Data

- `contour15sp.csv`, `contour20sp.csv`, `contour30sp.csv` - evenly spaced points (with normals) along the 15-, 20-, and 30-m depth contours, generated by `find_contour.ipynb` and used throughout the energy-flux notebooks.

## Notes

- Several notebooks contain hardcoded local paths (e.g. `F:/crs/...`, `C:/crs/...`) that will need to be updated for other environments.
- Some notebooks are duplicate/iterative versions of the same analysis (e.g. `WL_comparision.ipynb` vs. `WL_comp_new.ipynb`; `download_ndbc.ipynb` vs. `download_ndbc_final.ipynb`).
- A few notebooks (`Untitled3.ipynb`, `load_examine_various.ipynb`, `examine_hurrywave.ipynb`) are scratch/exploratory and not finished pipelines.
