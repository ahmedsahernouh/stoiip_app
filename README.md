# stoiip_app
STOIIP Monte-Carlo Streamlit Application
Volumetric Estimation with Uncertainty and ZMAP Grid Integration

This application provides an interactive Streamlit environment for estimating Stock Tank Oil Initially In Place (STOIIP) using a Monte-Carlo volumetric workflow. It supports both manual parameter entry and ingestion of ZMAP reservoir property grids, enabling spatially heterogeneous per-cell STOIIP calculations and map-based visualization.

-----------------------------------------------------------------------------------------------

## Features
Monte-Carlo Volumetric Simulation
STOIIP calculated from:
STOIIP = 7758 × Area(acres) × h(ft) × φ × So × NTG / Bo
Uncertainty applied through normal distributions.
Outputs include P10, P50, P90 and a tornado sensitivity chart.

-----------------------------------------------------------------------------------------------

## ZMAP Grid Support
Reads raw .dat ZMAP grids for structure, porosity, saturation, NTG, and thickness.
Corrects grid orientation and handles nulls.
Optional sign inversion for TVDSS.
All grids upscaled to the user-defined app cell size.

-----------------------------------------------------------------------------------------------

## Per-Cell STOIIP Mapping
Computes STOIIP for each active cell.
Active area determined from user-defined OWC (with uncertainty).
2D maps display structure, properties, and STOIIP.
3D PyVista view available with adjustable vertical exaggeration.

-----------------------------------------------------------------------------------------------

## Flexible Input Modes
Each reservoir parameter may come from:
Manual constant value
Manual value + uncertainty (synthetic heterogeneity)
ZMAP grid

-----------------------------------------------------------------------------------------------

## Running the Application
#### Install dependencies
pip install -r requirements.txt
or use Install_1st_Time.bat on Windows.
#### Launch Streamlit app
streamlit run app.py
or use run_app.bat on Windows.

-----------------------------------------------------------------------------------------------

## Repository Structure
###### app.py                 
Main Streamlit application
###### requirements.txt       
Python dependencies
###### run_app.bat            
Windows launcher
###### zmap_samples/          
Example ZMAP files
###### notebooks/             
Monte-Carlo explanation notebook
###### assets/                
Architecture diagram and images

-----------------------------------------------------------------------------------------------
## Licensing and Acknowledgement
This project extends the open-source work:
Original repository:
https://github.com/nadergerges/STOIIP_CALCULATIONS_WITH_UNCERTAINITIES
Author: Nader Gerges
License: GPL-3.0
Accordingly, this project is released under the GNU GPL-3.0 license.
