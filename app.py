"""STOIIP calculator with uncertainty and optional aligned ZMAP grids."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stoiip_core import Grid, read_zmap, require_aligned, weighted_volume, node_area_acres
from web_model import simulate

st.set_page_config(page_title='STOIIP Calculator | SaherLabs',layout='wide',page_icon='🛢️')
st.title('STOIIP Calculator')
st.caption('Estimate oil initially in place and explore uncertainty with scalar inputs or ZMAP grids.')
st.markdown('[SaherLabs](https://saherlabs.dev/projects) · [Source code](https://github.com/ahmedsahernouh/stoiip_app)')
st.latex(r'\mathrm{STOIIP\ (STB)}=7758\ A_{\mathrm{acres}}\ h_{\mathrm{ft}}\ \phi\ S_o\ \mathrm{NTG}/B_o')
mode=st.selectbox('Input mode',['Manual area','Synthetic grid example','Upload ZMAP grids'])
uploaded=None
property_uploads={}
confirmed=False
if mode=='Upload ZMAP grids':
    st.subheader('Select ZMAP grids')
    st.caption('Upload a structure grid to define the area. Add any property grids below; omitted properties use the scalar values in the sidebar. Bo uses a scalar value.')
    uploaded=st.file_uploader('Structure ZMAP (required)',type=['dat','zmap','grd'],key='grid_structure')
    columns=st.columns(2)
    for index,(name,label) in enumerate([('thickness','Thickness (ft)'),('porosity','Porosity (fraction)'),('saturation','Oil saturation (fraction)'),('ntg','Net-to-gross (fraction)')]):
        with columns[index%2]:
            property_uploads[name]=st.file_uploader(label+' ZMAP (optional)',type=['dat','zmap','grd'],key='grid_'+name)
    st.caption('All grids must share the same geometry. XY coordinates must be metres; thickness must be feet; porosity, saturation and NTG must be fractions. Uploads are processed in memory for your session.')
    confirmed=st.checkbox('My grid XY coordinates are in metres',value=False)
st.sidebar.header('Scalar inputs')
area=st.sidebar.number_input('Area (acres)',min_value=1.0,max_value=2_000_000.0,value=10000.0,step=100.0,disabled=mode!='Manual area')
base={}
for name,label,minimum,maximum,value,step in [('thickness','Thickness (ft)',0.1,5000.0,100.0,1.0),('porosity','Porosity (fraction)',0.001,1.0,0.2,0.01),('saturation','Oil saturation (fraction)',0.001,1.0,0.7,0.01),('ntg','Net-to-gross (fraction)',0.001,1.0,0.8,0.01),('bo','Formation volume factor Bo (rb/STB)',0.1,5.0,1.2,0.05)]:
    base[name]=st.sidebar.number_input(label,min_value=minimum,max_value=maximum,value=value,step=step)
unc={}
with st.sidebar.expander('Uncertainty inputs',expanded=True):
    st.caption('Relative standard deviation (1σ). Zero gives a constant input.')
    for name,label,default in [('area','Area',10.0),('thickness','Thickness',15.0),('porosity','Porosity',20.0),('saturation','Oil saturation',10.0),('ntg','Net-to-gross',10.0),('bo','Bo',5.0)]:
        unc[name]=st.number_input(label+' uncertainty (%)',min_value=0.0,max_value=50.0,value=default,step=1.0,disabled=name=='area' and mode!='Manual area',key='unc_'+name)
count=st.sidebar.number_input('Monte Carlo samples',min_value=100,max_value=10000,value=2000,step=100)
seed=st.sidebar.number_input('Random seed',min_value=0,max_value=2147483647,value=12345,step=1)
contact=None;contact_sigma=0;depth_positive_down=True
properties={k:base[k] for k in ('thickness','porosity','saturation','ntg')}
try:
    if mode=='Manual area':
        side=np.sqrt(area*4046.8564224);grid=Grid(np.zeros((30,30)),0.0,side,0.0,side)
    elif mode=='Synthetic grid example':
        x=np.linspace(0,3900,40);y=np.linspace(2900,0,30);xx,yy=np.meshgrid(x,y)
        grid=Grid(1500-160*np.exp(-((xx-1950)**2+(yy-1450)**2)/900000),0.0,3900.0,0.0,2900.0)
        st.info('Synthetic example: XY and depth in metres; property values use the sidebar inputs.')
    else:
        if uploaded is None:st.info('Select a structure ZMAP file above to calculate grid-based volumes.');st.stop()
        if not confirmed:st.info('Confirm the XY coordinate units above before calculating grid area.');st.stop()
        grid=read_zmap(uploaded.getvalue())
        for name,upload in property_uploads.items():
            if upload is not None:
                p=read_zmap(upload.getvalue());require_aligned(grid,p);properties[name]=p.values
    if mode!='Manual area':
        convention=st.sidebar.selectbox('Vertical convention',['Depth (positive down)','Elevation (positive up)'])
        depth_positive_down=convention.startswith('Depth')
        contact=st.sidebar.number_input('Oil-water contact (structure grid units)',value=1450.0 if mode=='Synthetic grid example' else 0.0)
        contact_sigma=st.sidebar.number_input('Contact uncertainty (1σ, grid units)',min_value=0.0,value=0.0,step=1.0)
        unc['area']=0
    _,_,volume_all=weighted_volume(grid,properties['thickness'],properties['porosity'],properties['saturation'],properties['ntg'],base['bo'])
    total,active_area,cells=weighted_volume(grid,properties['thickness'],properties['porosity'],properties['saturation'],properties['ntg'],base['bo'],contact=contact,depth_positive_down=depth_positive_down)
    if total<=0:st.warning('No positive oil volume is included by the current properties and contact. Adjust the inputs to continue.');st.stop()
    valid=np.isfinite(cells);weights=node_area_acres(grid);effective=dict(base)
    for name,value in properties.items():
        values=np.broadcast_to(value,grid.values.shape)
        effective[name]=float(np.sum(values[valid]*weights[valid])/np.sum(weights[valid]))
    samples,sensitivity=simulate(total,effective,unc,int(count),int(seed),structure=grid.values if contact is not None else None,volume_all=volume_all,contact=contact,contact_sigma=contact_sigma,depth_positive_down=depth_positive_down)
except ValueError as exc:
    st.error(str(exc));st.stop()
q10,q50,q90=np.percentile(samples,[10,50,90])/1e9
for col,label,value in zip(st.columns(4),['Deterministic (BSTB)','10th percentile (low)','50th percentile','90th percentile (high)'],[total/1e9,q10,q50,q90]):col.metric(label,f'{value:,.4f}')
st.caption(f'Included area: {active_area:,.2f} acres · Volumes in billions of stock-tank barrels (BSTB). Percentile labels use cumulative probability.')
left,right=st.columns(2)
with left:
    st.subheader('STOIIP distribution')
    hist=go.Figure(go.Histogram(x=samples/1e9,nbinsx=45,marker_color='#168b8b'))
    for value,label in [(q10,'10th'),(q50,'50th'),(q90,'90th')]:hist.add_vline(x=value,line_dash='dash',annotation_text=label)
    hist.update_layout(xaxis_title='STOIIP (BSTB)',yaxis_title='Samples',margin=dict(t=35,l=10,r=10,b=10));st.plotly_chart(hist,width='stretch')
with right:
    st.subheader('One-factor sensitivity')
    labels={'area':'Area / contact','thickness':'Thickness','porosity':'Porosity','saturation':'Oil saturation','ntg':'Net-to-gross','bo':'Bo'}
    ordered=sorted(sensitivity,key=sensitivity.get)
    fig=go.Figure(go.Bar(x=[100*sensitivity[k] for k in ordered],y=[labels[k] for k in ordered],orientation='h',marker_color='#378add'))
    fig.update_layout(xaxis_title='Relative one-factor impact (%)',margin=dict(t=35,l=10,r=10,b=10));st.plotly_chart(fig,width='stretch')
st.subheader('Reservoir property view')
prop=st.selectbox('Property',['Node STOIIP contribution (MSTB)','Thickness (ft)','Porosity','Oil saturation','Net-to-gross','Structure'])
lookup={'Node STOIIP contribution (MSTB)':cells/1e6,'Thickness (ft)':properties['thickness'],'Porosity':properties['porosity'],'Oil saturation':properties['saturation'],'Net-to-gross':properties['ntg'],'Structure':grid.values}
color=np.where(valid,np.broadcast_to(lookup[prop],grid.values.shape),np.nan)
view=st.radio('View',['Map','3D surface'],horizontal=True)
if view=='Map':
    fig=go.Figure(go.Heatmap(x=grid.x,y=grid.y,z=color,colorscale='Viridis',colorbar=dict(title=prop)))
    fig.update_layout(xaxis_title='X (m)',yaxis_title='Y (m)',margin=dict(t=20,l=10,r=10,b=10))
else:
    xx,yy=np.meshgrid(grid.x,grid.y)
    fig=go.Figure(go.Surface(x=xx,y=yy,z=np.where(valid,grid.values,np.nan),surfacecolor=color,colorscale='Viridis',colorbar=dict(title=prop)))
    fig.update_layout(scene=dict(xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Structure (input units)' if contact is not None else 'Flat reference surface',zaxis=dict(autorange='reversed' if depth_positive_down else True)),margin=dict(t=20,l=10,r=10,b=10))
st.plotly_chart(fig,width='stretch')
st.download_button('Download Monte Carlo samples',pd.DataFrame({'STOIIP_STB':samples}).to_csv(index=False),'stoiip-samples.csv','text/csv')
with st.expander('Calculation method'):
    st.write('Scalar inputs use area in acres, thickness in feet, and fractional porosity, oil saturation and net-to-gross. Grid XY coordinates must be metres. ZMAP node values use trapezoidal edge weights; the contact masks nodes according to the vertical convention. Property grids must match the structure geometry.')
    st.write('Uncertainties use clipped normal distributions and a reproducible random seed. Property uncertainties are global multipliers; contact uncertainty integrates nodal volume at each sampled contact. Sensitivity compares one-factor standard deviations, not variance allocations. The property view shows deterministic inputs.')
    st.caption('Illustrative volumetric analysis. Results depend on the supplied data and assumptions.')
