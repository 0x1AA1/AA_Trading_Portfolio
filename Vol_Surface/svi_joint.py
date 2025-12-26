"""
SVI Joint Calibration with Static-Arbitrage Penalties (simplified)

Fits raw SVI per expiry and adds penalties to discourage butterfly (convexity)
violations across strikes and calendar monotonicity across expiries.

Note: This is a simplified penalty approach built on top of per-expiry SVI.
For production, consider convex optimization frameworks (e.g., cvxpy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.optimize import minimize  # type: ignore


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi_joint(options_df: pd.DataFrame, spot: float) -> Dict:
    df = options_df.copy()
    if df.empty:
        return {'status':'empty'}
    df = df[np.isfinite(df['impliedVolatility']) & (df['impliedVolatility']>0)]
    if df.empty:
        return {'status':'no_iv'}
    df['expiration'] = pd.to_datetime(df['expiration'])
    expiries = sorted(df['expiration'].unique())
    # Initial per-expiry fits: use ATM IV to set starting variance
    init_params = []
    data_per_exp = []
    for exp in expiries:
        g = df[df['expiration']==exp]
        if g.empty:
            continue
        T = max(((pd.Timestamp(exp) - pd.Timestamp.now()).days)/365.0, 1/365)
        strikes = g['strike'].values.astype(float)
        k = np.log(np.maximum(strikes, 1e-6)/max(spot,1e-6))
        iv = g['impliedVolatility'].values.astype(float)
        atm_var = float(np.nanmedian((iv**2)*T)) if len(iv) else 0.04
        init_params.extend([atm_var, 0.1, 0.0, 0.0, 0.1])
        data_per_exp.append((T, k, iv))
    if not data_per_exp:
        return {'status':'no_data'}

    def bounded(x):
        out = []
        for i in range(0,len(x),5):
            a,b,rho,m,sig = x[i:i+5]
            b = max(b,1e-8)
            sig = max(sig,1e-8)
            rho = float(np.clip(rho, -0.999, 0.999))
            out.extend([a,b,rho,m,sig])
        return np.array(out)

    def loss(theta: np.ndarray) -> float:
        th = bounded(theta)
        err = 0.0
        # Fit error per expiry
        idx = 0
        for (T,k,iv) in data_per_exp:
            a,b,rho,m,sig = th[idx:idx+5]
            w = svi_total_variance(k,a,b,rho,m,sig)
            w = np.maximum(w,1e-10)
            model_iv = np.sqrt(w/max(T,1e-8))
            e = model_iv - iv
            err += float(np.nanmean(e**2))
            idx += 5
        # Penalties: convexity across strikes (second derivative >= 0)
        pen = 0.0
        idx = 0
        for (T,k,iv) in data_per_exp:
            a,b,rho,m,sig = th[idx:idx+5]
            ks = np.linspace(np.nanmin(k), np.nanmax(k), 15)
            w = svi_total_variance(ks,a,b,rho,m,sig)
            # discrete second difference
            sec = w[:-2] - 2*w[1:-1] + w[2:]
            pen += float(np.sum(np.minimum(sec,0)**2))
            idx += 5
        # Calendar monotonicity: total variance increases with T at same k~0
        # Rough check at k=0
        if len(data_per_exp) >= 2:
            wT = []
            idx = 0
            for (T,_,_) in data_per_exp:
                a,b,rho,m,sig = th[idx:idx+5]
                w0 = svi_total_variance(np.array([0.0]),a,b,rho,m,sig)[0]
                wT.append((T,w0))
                idx += 5
            wT = sorted(wT, key=lambda x:x[0])
            for i in range(len(wT)-1):
                if wT[i+1][1] < wT[i][1]:
                    pen += (wT[i][1] - wT[i+1][1])**2
        return err + 1e-3*pen

    res = minimize(loss, np.array(init_params), method='L-BFGS-B')
    th = bounded(res.x)
    out = {'status':'ok','params':[], 'qc':{'rmse':None, 'penalty':None}}
    # Compute RMSE summary
    rmse = 0.0; n=0
    idx=0
    for (T,k,iv) in data_per_exp:
        a,b,rho,m,sig = th[idx:idx+5]
        w = svi_total_variance(k,a,b,rho,m,sig)
        w = np.maximum(w,1e-10)
        model_iv = np.sqrt(w/max(T,1e-8))
        e = model_iv - iv
        rmse += float(np.nanmean(e**2)); n+=1
        out['params'].append({'T':T,'a':a,'b':b,'rho':rho,'m':m,'sigma':sig,'rmse':float(np.sqrt(np.nanmean(e**2)))})
        idx+=5
    out['qc']['rmse'] = float(np.sqrt(rmse/max(n,1)))
    return out

