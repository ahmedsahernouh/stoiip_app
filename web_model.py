"""Bounded Monte Carlo helpers for the STOIIP application."""
import numpy as np

def sample(base, sigma_percent, count, rng, lower=0.0, upper=None):
    if sigma_percent == 0:
        return np.full(count, base, dtype=float)
    return np.clip(rng.normal(base, abs(base)*sigma_percent/100, count), lower, upper)

def contact_totals(structure, volume, contacts, depth_positive_down):
    valid=np.isfinite(structure)&np.isfinite(volume)
    order=np.argsort(structure[valid]);depths=structure[valid][order]
    cumulative=np.concatenate(([0.0],np.cumsum(volume[valid][order])))
    if depth_positive_down:
        return cumulative[np.searchsorted(depths,contacts,side='right')]
    return cumulative[-1]-cumulative[np.searchsorted(depths,contacts,side='left')]

def simulate(base_volume, base_properties, uncertainties, count, seed, *, structure=None,
             volume_all=None, contact=None, contact_sigma=0, depth_positive_down=True):
    rng=np.random.default_rng(seed);factors={}
    for name in ('thickness','porosity','saturation','ntg','bo'):
        base=base_properties[name]
        upper=1.0 if name in ('porosity','saturation','ntg') else None
        draws=sample(base,uncertainties[name],count,rng,0.01 if name=='bo' else 0.0,upper)
        factors[name]=draws/base if name!='bo' else base/draws
    if structure is not None and contact_sigma>0:
        draws=rng.normal(contact,contact_sigma,count)
        factors['area']=contact_totals(structure,volume_all,draws,depth_positive_down)/base_volume
    else:factors['area']=sample(1,uncertainties['area'],count,rng,0)
    combined=np.ones(count)
    for factor in factors.values():combined*=factor
    impacts={k:float(np.std(base_volume*v)) if np.ptp(v)>0 else 0.0 for k,v in factors.items()}
    total=sum(impacts.values())
    return base_volume*combined,{k:v/total if total else 0 for k,v in impacts.items()}
