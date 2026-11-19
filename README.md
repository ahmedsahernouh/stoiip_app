# STOIIP Calculator

[Open the web app](https://api.saherlabs.dev/stoiip/)

A Streamlit calculator for oil initially in place, uncertainty analysis and ZMAP-based reservoir property review.

## Features

- Scalar area, thickness, porosity, oil saturation, net-to-gross and formation volume factor.
- Reproducible Monte Carlo distributions, cumulative percentiles and one-factor sensitivity.
- Structure and aligned property ZMAP grids, with explicit unit and vertical-convention inputs.
- Map and interactive 3D views, plus CSV export of simulation samples.
- A synthetic grid example that runs without external data.

## Select grids

Choose **Upload ZMAP structure grid** from **Area / structure source** to define reservoir geometry. Under **Property inputs**, select **Manual value** or **ZMAP grid** independently for thickness, porosity, oil saturation and net-to-gross. Each property set to ZMAP grid shows its own file selector and requires a file; the other properties retain their manual values. Bo is scalar. Confirm metre XY coordinates and set the vertical convention and oil-water contact above the property inputs.

## Run locally

```sh
python -m venv .venv
# Activate the virtual environment, then:
pip install -r requirements.txt
streamlit run app.py --server.baseUrlPath "" --browser.serverAddress localhost --browser.serverPort 8501
```

The checked-in Streamlit configuration also supports deployment behind an HTTPS proxy at `/stoiip/`.

## Calculation model

The volumetric formula uses area in acres, thickness in feet and dimensionless fractional properties. Grid XY coordinates must be metres. ZMAP node values are integrated using trapezoidal boundary weights; uploaded property grids must match the structure geometry. The node mask follows the selected depth/elevation convention and oil-water contact.

Uncertainties are clipped normal distributions. Property uncertainty acts through global multipliers; contact uncertainty reintegrates the included nodal volume. Sensitivity compares one-factor standard deviations, not a statistical variance decomposition. Percentile labels explicitly use cumulative probability.

Uploads are limited to 8 MiB and 100,000 nodes. Grid results are a nodal approximation whose accuracy depends on resolution and input assumptions. This is an illustrative calculation tool, not a reserves certification.

## Verification

```sh
python -m unittest discover -s tests -v
```

The tests cover unit conversion, analytic volumes, grid orientation/alignment, nulls, invalid inputs, contact selection and simulation reproducibility.

## Notebook study

Earlier Monte Carlo analysis remains in [notebooks](notebooks/). The old notebooks refer to external inputs not included in the repository. Existing `zmap_samples` are historical study material and are not the app's synthetic example.

## License

GNU GPL v3; see [LICENSE](LICENSE).
