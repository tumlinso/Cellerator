#!/usr/bin/env python3
"""Compare scalar, compiled CPU and CUDA velocity precision paths.

Every backend consumes the same serialized stored S/U/W bytes and masks.
Input quantization effects are reported separately from backend differences.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, issparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import qualify_stochastic as qs  # pinned source loaders and existing M2 oracle methods

SCVELO = qs.SCVELO
DTYPE_CODE = {np.dtype('float16'): 1, np.dtype('float32'): 2, np.dtype('float64'): 3}
CODE_DTYPE = {v: k for k, v in DTYPE_CODE.items()}
METHOD_CODE = {1: 'deterministic', 2: 'stochastic'}
PRECISION_GATES = {
    'f32': {'moment_atol': 2e-6, 'moment_rtol': 2e-5,
            'value_atol': 2e-5, 'value_rtol': 2e-4,
            'r2_atol': 2e-4, 'objective_normalized': 5e-4},
    'f64': {'moment_atol': 1e-12, 'moment_rtol': 1e-11,
            'value_atol': 1e-10, 'value_rtol': 1e-9,
            'r2_atol': 1e-10, 'objective_normalized': 1e-9},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dense32(x):
    if issparse(x):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def fixture_arrays(name: str):
    path = SCVELO / 'tests/_data' / f'{name}_100obs_preprocessed.h5ad'
    s, u, w, hvg = qs.fixture(path)
    return path, dense32(s), dense32(u), dense32(w), np.asarray(hvg, dtype=bool)


def storage_profiles():
    return {
        'fp32': (np.dtype('float32'), np.dtype('float32'), np.dtype('float32')),
        'fp64': (np.dtype('float64'), np.dtype('float64'), np.dtype('float64')),
        'fp16': (np.dtype('float16'), np.dtype('float16'), np.dtype('float16')),
        'mixed_s16_u32_w64': (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')),
        'mixed_s64_u16_w32': (np.dtype('float64'), np.dtype('float16'), np.dtype('float32')),
    }


def quantize_inputs(s, u, w, profile):
    dtypes = storage_profiles()[profile]
    return tuple(np.asarray(x, dtype=t) for x, t in zip((s, u, w), dtypes))


def write_input(path: Path, arrays, fit_mask, gene_mask):
    s, u, w = arrays
    if s.ndim != 2 or u.shape != s.shape or w.shape != (s.shape[0], s.shape[0]):
        raise ValueError('fixture matrix shapes do not agree')
    n, g = s.shape
    with path.open('wb') as f:
        f.write(b'VPRCIN1\0')
        f.write(struct.pack('<QQ', n, g))
        f.write(bytes(DTYPE_CODE[np.dtype(x.dtype)] for x in (s, u, w)))
        f.write(bytes(5))
        for x in (s, u, w):
            dt = np.dtype(x.dtype).newbyteorder('<')
            f.write(np.ascontiguousarray(x, dtype=dt).tobytes(order='C'))
        f.write(np.ascontiguousarray(fit_mask, dtype=np.uint8).tobytes())
        f.write(np.ascontiguousarray(gene_mask, dtype=np.uint8).tobytes())
    return sha256(path.read_bytes())


def read_output(path: Path, n: int, g: int):
    with path.open('rb') as f:
        if f.read(8) != b'VPRCOT1\0':
            raise RuntimeError('bad precision output magic')
        tags = np.fromfile(f, dtype='u1', count=5)
        dims = np.fromfile(f, dtype='<u8', count=2)
        if tags.size != 5 or not np.array_equal(dims, [n, g]):
            raise RuntimeError('invalid precision output header')
        precision = 'f32' if tags[0] == 1 else 'f64' if tags[0] == 2 else None
        if precision is None or int(tags[1]) not in METHOD_CODE:
            raise RuntimeError('invalid output precision or method tag')
        dt = '<f4' if precision == 'f32' else '<f8'

        def get(dtype, shape):
            count = int(np.prod(shape, dtype=np.int64))
            result = np.fromfile(f, dtype=dtype, count=count)
            if result.size != count:
                raise RuntimeError('truncated precision output')
            return result.reshape(shape)

        out = dict(precision=precision, method=METHOD_CODE[int(tags[1])],
                   low_signal=bool(tags[2]), refit_all=bool(tags[3]),
                   backend_code=int(tags[4]))
        for key in ('ms', 'mu', 'mss', 'mus', 'velocity', 'second_residual'):
            out[key] = get(dt, (n, g))
        for key in ('gamma', 'offset', 'offset2', 'first_scale', 'second_scale',
                    'r2', 'objective', 'initial_gamma', 'initial_offset'):
            out[key] = get(dt, (g,))
        for key in ('selected', 'initial_status', 'status'):
            out[key] = get('u1', (g,))
        if f.read(1):
            raise RuntimeError('trailing precision output bytes')
        return out


def oracle_moments(s, u, w, dtype):
    """Independent SciPy sparse products on the exact stored values."""
    dtype = np.dtype(dtype)
    ss = csr_matrix(np.asarray(s, dtype=dtype)); uu = csr_matrix(np.asarray(u, dtype=dtype))
    ww = csr_matrix(np.asarray(w, dtype=dtype))
    for x in (ss, uu, ww):
        x.sum_duplicates(); x.sort_indices()
    ms = np.asarray((ww @ ss).toarray(), dtype=dtype)
    mu = np.asarray((ww @ uu).toarray(), dtype=dtype)
    mss = np.asarray((ww @ ss.multiply(ss)).toarray(), dtype=dtype)
    mus = np.asarray((ww @ ss.multiply(uu)).toarray(), dtype=dtype)
    return ms, mu, mss, mus


def fp64_joint_lstsq(ms, mu, mss, mus, fit_mask, columns, first_scale,
                     second_scale, fit_offset, fit_offset2, trimming):
    """Independent FP64 QR/SVD solve of the declared joint weighted objective."""
    var=2*mss-ms;cov=2*mus+mu;rows=np.flatnonzero(fit_mask)
    gamma=np.zeros(len(columns),np.float64);offset=np.zeros_like(gamma)
    offset2=np.zeros_like(gamma);objective=np.zeros_like(gamma)
    for k,j in enumerate(columns):
        if rows.size==0: continue
        s=np.asarray(ms[rows,j],np.float64);u=np.asarray(mu[rows,j],np.float64)
        if trimming=='none': keep=np.ones(rows.size,dtype=bool)
        else:
            max_s=max(float(np.max(s)),1e-3);max_u=max(float(np.max(u)),1e-3)
            joint=s/max_s+u/max_u
            qlo,qhi=np.percentile(joint,(5,95),method='linear')
            slo,shi=np.percentile(s,(5,95),method='linear')
            if trimming=='tails' and fit_offset:
                keep=(joint<=qlo)|(joint>=qhi)|(s<=slo)|(s>=shi)
            else:
                keep=(joint>=qhi)|(s>=shi)
        a=np.where(keep,s,0.);b=np.where(keep,u,0.)
        c=np.asarray(var[rows,j],np.float64);d=np.asarray(cov[rows,j],np.float64)
        scale1=float(first_scale[k]);scale2=float(second_scale[k])
        design=np.zeros((2*rows.size,1+int(fit_offset)+int(fit_offset2)),np.float64)
        target=np.concatenate((b/scale1,d/scale2))
        design[:rows.size,0]=a/scale1;design[rows.size:,0]=c/scale2
        col=1
        if fit_offset:
            design[:rows.size,col]=1./scale1;col+=1
        if fit_offset2:
            design[rows.size:,col]=1./scale2
        solution,*_=np.linalg.lstsq(design,target,rcond=None)
        gamma[k]=solution[0]
        col=1
        if fit_offset: offset[k]=solution[col];col+=1
        if fit_offset2: offset2[k]=solution[col]
        r1=(b-gamma[k]*a-offset[k])/scale1
        r2=(d-gamma[k]*c-offset2[k])/scale2
        objective[k]=float(np.dot(r1,r1)+np.dot(r2,r2))/(2.*rows.size)
    return dict(gamma=gamma,offset=offset,offset2=offset2,objective=objective,
                var_ss=var[np.ix_(rows,columns)],cov_us=cov[np.ix_(rows,columns)])


def independent_oracle(ns, stored, fit_mask, gene_mask, compute, method,
                       fit_offset, fit_offset2, trimming, min_r2=.01, min_ratio=.01):
    s, u, w = stored
    ms, mu, mss, mus = oracle_moments(s, u, w, compute)
    percentile = (5, 95) if trimming == 'tails' else None
    init = qs._source_initial(ns, ms, mu, fit_mask, gene_mask,
                              fit_offset, percentile)
    selected = qs._native_selection(init, ms, mu, gene_mask, min_r2, min_ratio)
    g = ms.shape[1]
    out = dict(ms=ms, mu=mu, mss=mss, mus=mus,
               gamma=np.asarray(init._gamma).copy(), offset=np.asarray(init._offset).copy(),
               offset2=np.zeros(g, dtype=compute), first_scale=np.zeros(g, dtype=compute),
               second_scale=np.zeros(g, dtype=compute), r2=np.asarray(init._r2).copy(),
               objective=np.zeros(g, dtype=compute), initial_gamma=np.asarray(init._gamma).copy(),
               initial_offset=np.asarray(init._offset).copy(), selected=selected.astype(np.uint8),
               velocity=np.asarray(init._residual).copy(),
               second_residual=np.zeros(ms.shape, dtype=compute))
    if method == 'deterministic':
        return out
    rows = np.flatnonzero(fit_mask)
    cols = np.flatnonzero(selected)
    if not cols.size:
        return out
    with np.errstate(all='ignore'):
        fit = qs._source_gls(ns, ms[np.ix_(rows, cols)], mu[np.ix_(rows, cols)],
                             mss[np.ix_(rows, cols)], mus[np.ix_(rows, cols)],
                             np.asarray(init._residual)[np.ix_(rows, cols)],
                             fit_offset, fit_offset2, percentile)
    if np.dtype(compute)==np.dtype('float64'):
        # The pinned source GLS returns FP32 coefficient fields in this route.
        # Re-solve its declared weighted joint objective independently in FP64
        # with NumPy's LAPACK least-squares path; retain independently measured
        # residual scales from the source definitions above.
        precise=fp64_joint_lstsq(ms,mu,mss,mus,fit_mask,cols,
            np.asarray(fit['residual_scale']),np.asarray(fit['residual2_scale']),
            fit_offset,fit_offset2,trimming)
        fit.update(precise)
    out['gamma'][cols] = fit['gamma']; out['offset'][cols] = fit['offset']
    out['offset2'][cols] = fit['offset2']
    out['first_scale'][cols] = fit['residual_scale']
    out['second_scale'][cols] = fit['residual2_scale']
    out['velocity'] = mu - out['gamma'][None, :] * ms
    if fit_offset:
        out['velocity'] -= out['offset'][None, :]
    var = 2*mss - ms; cov = 2*mus + mu
    r2 = cov[:, cols] - 2*ms[:, cols]*mu[:, cols]
    r2 -= out['gamma'][None, cols] * (var[:, cols] - 2*ms[:, cols]**2)
    if fit_offset:
        r2 += 2*out['offset'][None, cols]*ms[:, cols]
    if fit_offset2:
        r2 -= out['offset2'][None, cols]
    out['second_residual'][:, cols] = r2
    out['objective'][cols] = qs.weighted_objective(
        ms[np.ix_(rows, cols)], mu[np.ix_(rows, cols)], fit,
        np.arange(cols.size), fit_offset, fit_offset2, percentile)
    return out


def quantile(values, q):
    values = np.asarray(values)
    if not values.size:
        return float('nan')
    return float(np.percentile(values, q, method='linear'))


def trim_state(ms, mu, fit_mask, fit_offset, trimming):
    """Return initial-fit and joint-fit row masks plus explicit boundary data."""
    rows = np.flatnonzero(fit_mask)
    g = ms.shape[1]
    initial_keep = np.zeros((ms.shape[0], g), dtype=bool)
    joint_keep = np.zeros((ms.shape[0], g), dtype=bool)
    boundaries = []
    for j in range(g):
        x = np.asarray(ms[rows, j]); y = np.asarray(mu[rows, j])
        if not rows.size:
            boundaries.append(dict(gene=j, fitting_rows=0))
            continue
        if trimming == 'none':
            initial_keep[rows, j] = True; joint_keep[rows, j] = True
            boundaries.append(dict(gene=j, fitting_rows=int(rows.size), trimming='none'))
            continue
        maxx=max(float(np.max(x)),1e-3); maxy=max(float(np.max(y)),1e-3)
        score=x/maxx+y/maxy
        qlo,qhi=quantile(score,5),quantile(score,95)
        lower=trimming=='tails' and fit_offset
        init=(score<=qlo)|(score>=qhi) if lower else score>=qhi
        joint_score=score
        spliced=np.asarray(x)
        slo,shi=quantile(spliced,5),quantile(spliced,95)
        joint=(joint_score<=qlo)|(joint_score>=qhi) if lower else joint_score>=qhi
        joint|=((spliced<=slo)|(spliced>=shi)) if lower else (spliced>=shi)
        initial_keep[rows,j]=init; joint_keep[rows,j]=joint
        eps=np.finfo(np.asarray(ms).dtype).eps
        boundaries.append(dict(
            gene=j, fitting_rows=int(rows.size), q05=qlo, q95=qhi,
            q05_rank=(rows.size-1)*.05, q95_rank=(rows.size-1)*.95,
            q05_ties=int(np.count_nonzero(np.abs(score-qlo)<=eps*max(1.,abs(qlo)))),
            q95_ties=int(np.count_nonzero(np.abs(score-qhi)<=eps*max(1.,abs(qhi)))),
            q05_inside_gap=float(np.min(np.abs(score[score>qlo]-qlo))) if np.any(score>qlo) else None,
            q95_inside_gap=float(np.min(np.abs(score[score<qhi]-qhi))) if np.any(score<qhi) else None,
            initial_boundary_rows=int(np.count_nonzero(init)),
            joint_boundary_rows=int(np.count_nonzero(joint))))
    return initial_keep,joint_keep,boundaries


def conditions(ms, mu, fit_mask, fit_offset, fit_offset2, trimming, method, result):
    rows=np.flatnonzero(fit_mask); g=ms.shape[1]
    init_condition=np.full(g,np.inf); joint_condition=np.full(g,np.inf)
    initial_keep,joint_keep,_=trim_state(ms,mu,fit_mask,fit_offset,trimming)
    for j in range(g):
        if not rows.size: continue
        take=rows[initial_keep[rows,j]]
        x=np.asarray(ms[take,j],dtype=np.float64)
        if x.size:
            design=np.column_stack((x,np.ones(x.size))) if fit_offset else x[:,None]
            try: init_condition[j]=float(np.linalg.cond(design))
            except np.linalg.LinAlgError: pass
        if method!='stochastic': continue
        if 'status' in result and result['status'][j]!=0: continue
        if 'status' not in result and not bool(result['selected'][j]): continue
        s1=float(result['first_scale'][j]);s2=float(result['second_scale'][j])
        if not (s1>0 and s2>0 and np.isfinite(s1+s2)): continue
        take=rows
        a=np.asarray(ms[take,j],dtype=np.float64).copy()
        b=np.asarray(mu[take,j],dtype=np.float64).copy()
        keep=joint_keep[take,j]
        a[~keep]=0.;b[~keep]=0.
        c=2*np.asarray(result['mss'][take,j],dtype=np.float64)-np.asarray(ms[take,j],dtype=np.float64)
        cols=1+int(fit_offset)+int(fit_offset2)
        design=np.zeros((2*take.size,cols),dtype=np.float64)
        design[:take.size,0]=a/s1;design[take.size:,0]=c/s2
        col=1
        if fit_offset:
            design[:take.size,col]=1/s1;col+=1
        if fit_offset2:
            design[take.size:,col]=1/s2
        try: joint_condition[j]=float(np.linalg.cond(design))
        except np.linalg.LinAlgError: pass
    return init_condition,joint_condition


def boundary_summary(ref, got, fit_mask, fit_offset, trimming):
    if trimming=='none':
        return dict(trimming='none',rank_comparison='not_applicable')
    a0,a1,ab=trim_state(ref['ms'],ref['mu'],fit_mask,fit_offset,trimming)
    b0,b1,bb=trim_state(got['ms'],got['mu'],fit_mask,fit_offset,trimming)
    rows=np.flatnonzero(fit_mask)
    max_shift=0; changed=0; gene_rows=[]
    for j in range(ref['ms'].shape[1]):
        if not rows.size: continue
        def ranks(result):
            x=result['ms'][rows,j]; y=result['mu'][rows,j]
            score=x/max(float(np.max(x)),1e-3)+y/max(float(np.max(y)),1e-3)
            order=np.argsort(score,kind='stable'); rank=np.empty(order.size,dtype=np.int64);rank[order]=np.arange(order.size)
            return rank
        shift=np.abs(ranks(ref)-ranks(got)); local=int(np.max(shift,initial=0))
        max_shift=max(max_shift,local)
        diff=int(np.count_nonzero(a0[rows,j]!=b0[rows,j]))
        changed+=diff
        if local or diff:
            gene_rows.append(dict(gene=j,max_rank_shift=local,initial_trim_membership_differences=diff,
                reference_q05=ab[j].get('q05'),reference_q95=ab[j].get('q95'),
                candidate_q05=bb[j].get('q05'),candidate_q95=bb[j].get('q95'),
                reference_q05_ties=ab[j].get('q05_ties'),reference_q95_ties=ab[j].get('q95_ties'),
                candidate_q05_ties=bb[j].get('q05_ties'),candidate_q95_ties=bb[j].get('q95_ties')))
    def boundary_aggregate(items):
        valid=[x for x in items if 'q05' in x and 'q95' in x]
        if not valid: return dict(genes=0)
        qlo=np.asarray([x['q05'] for x in valid]);qhi=np.asarray([x['q95'] for x in valid])
        return dict(genes=len(valid),q05_min=float(np.min(qlo)),q05_median=float(np.median(qlo)),
            q05_max=float(np.max(qlo)),q95_min=float(np.min(qhi)),q95_median=float(np.median(qhi)),
            q95_max=float(np.max(qhi)),genes_with_q05_ties=int(sum(x.get('q05_ties',0)>1 for x in valid)),
            genes_with_q95_ties=int(sum(x.get('q95_ties',0)>1 for x in valid)),
            maximum_tie_count=max(max(x.get('q05_ties',0),x.get('q95_ties',0)) for x in valid),
            minimum_inside_gap_q05=min((x['q05_inside_gap'] for x in valid if x.get('q05_inside_gap') is not None),default=None),
            minimum_inside_gap_q95=min((x['q95_inside_gap'] for x in valid if x.get('q95_inside_gap') is not None),default=None))
    qlo_delta=[abs(x['q05']-y['q05']) for x,y in zip(ab,bb) if 'q05' in x and 'q05' in y]
    qhi_delta=[abs(x['q95']-y['q95']) for x,y in zip(ab,bb) if 'q95' in x and 'q95' in y]
    return dict(trimming=trimming,reference_boundary_quantiles=boundary_aggregate(ab),
         candidate_boundary_quantiles=boundary_aggregate(bb),
         q05_threshold_max_abs_delta=float(max(qlo_delta,default=0.)),
         q95_threshold_max_abs_delta=float(max(qhi_delta,default=0.)),
         genes_with_quantile_boundary_ties_reference=int(sum(
             (x.get('q05_ties',0)>1 or x.get('q95_ties',0)>1) for x in ab)),
         genes_with_quantile_boundary_ties_candidate=int(sum(
             (x.get('q05_ties',0)>1 or x.get('q95_ties',0)>1) for x in bb)),
         row_rank_max_shift=max_shift, initial_trim_membership_differences=changed,
         affected_genes=gene_rows[:10])


def field_gate(got, ref, atol, rtol, mask=None, absolute=False, normalized=False):
    got=np.asarray(got,dtype=np.float64);ref=np.asarray(ref,dtype=np.float64)
    if mask is not None and got.ndim==1:
        got=got[mask];ref=ref[mask]
    elif mask is not None and got.ndim==2:
        got=got[:,mask];ref=ref[:,mask]
    if not got.size:
        return dict(pass_gate=True,compared=0,max_normalized_error=0.,max_absolute_error=0.)
    finite=np.isfinite(got)&np.isfinite(ref)
    if not np.all(finite):
        return dict(pass_gate=False,compared=int(got.size),nonfinite=int(got.size-np.count_nonzero(finite)),
                    max_normalized_error=float('inf'),max_absolute_error=float('inf'))
    diff=np.abs(got-ref)
    if absolute:
        norm=diff/max(float(atol),np.finfo(np.float64).tiny)
        ok=bool(np.all(diff<=atol))
    elif normalized:
        scale=np.maximum(1.,np.abs(ref))
        norm=diff/(float(atol)*scale)
        ok=bool(np.all(diff<=float(atol)*scale))
    else:
        if ref.ndim==2:
            scale=np.maximum(1.,np.max(np.abs(ref),axis=0,keepdims=True))
        else:
            scale=np.maximum(1.,np.abs(ref))
        limit=float(atol)*scale+float(rtol)*np.abs(ref)
        norm=diff/np.maximum(limit,np.finfo(np.float64).tiny)
        ok=bool(np.all(diff<=limit))
    return dict(pass_gate=ok,compared=int(got.size),max_normalized_error=float(np.max(norm)),
                max_absolute_error=float(np.max(diff)))


def compare_results(got, ref, model, precision, condition, *, mask_checks=True):
    gates=PRECISION_GATES[precision]; checks={}
    moments=('ms','mu') if model=='deterministic' else ('ms','mu','mss','mus')
    for key in moments:
        checks[key]=field_gate(got[key],ref[key],gates['moment_atol'],gates['moment_rtol'])
    if model=='deterministic':
        init_valid=(condition['initial']<=1e3)&np.isfinite(condition['initial'])
        for key in ('gamma','offset','initial_gamma','initial_offset'):
            checks[key]=field_gate(got[key],ref[key],gates['value_atol'],gates['value_rtol'],init_valid)
        checks['velocity']=field_gate(got['velocity'],ref['velocity'],gates['value_atol'],gates['value_rtol'])
        checks['r2']=field_gate(got['r2'],ref['r2'],gates['r2_atol'],0,absolute=True)
        branches={'initial_status_exact':bool(np.array_equal(got['initial_status'],ref['initial_status'])) if 'initial_status' in ref else None,
                  'status_exact':bool(np.array_equal(got['status'],ref['status'])) if 'status' in ref else None}
    else:
        valid=(got['status']==0)&(condition['joint']<=1e3)&np.isfinite(condition['joint'])
        init_valid=(got['initial_status']==0)&(condition['initial']<=1e3)&np.isfinite(condition['initial'])
        for key in ('gamma','offset','offset2'):
            checks[key]=field_gate(got[key],ref[key],gates['value_atol'],gates['value_rtol'],valid)
        for key in ('first_scale','second_scale'):
            checks[key]=field_gate(got[key],ref[key],gates['value_atol'],gates['value_rtol'],valid)
        for key in ('initial_gamma','initial_offset'):
            checks[key]=field_gate(got[key],ref[key],gates['value_atol'],gates['value_rtol'],init_valid)
        checks['velocity']=field_gate(got['velocity'],ref['velocity'],gates['value_atol'],gates['value_rtol'])
        checks['second_residual']=field_gate(got['second_residual'],ref['second_residual'],gates['value_atol'],gates['value_rtol'])
        checks['r2']=field_gate(got['r2'],ref['r2'],gates['r2_atol'],0,absolute=True)
        checks['objective']=field_gate(got['objective'],ref['objective'],gates['objective_normalized'],0,normalized=True,mask=valid)
        branches={'selected_exact':bool(np.array_equal(got['selected'],ref['selected'])) if 'selected' in ref else None,
                  'status_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(got['status'],return_counts=True))},
                  'initial_status_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(got['initial_status'],return_counts=True))}}
    if model=='deterministic' and 'selected' in ref:
        branches['selected_exact']=bool(np.array_equal(got['selected'],ref['selected']))
    branches['parameter_condition_max']=float(np.max(condition['initial'],initial=0.))
    branches['parameter_condition_excluded_genes']=int(np.count_nonzero(~(condition['initial']<=1e3)))
    if model=='stochastic':
        branches['joint_condition_max']=float(np.max(condition['joint'],initial=0.))
        branches['joint_condition_excluded_genes']=int(np.count_nonzero(~(condition['joint']<=1e3)))
    branches['all_numerical_fields_finite']=all(bool(np.isfinite(got[k]).all()) for k in
        ('ms','mu','mss','mus','velocity','second_residual','gamma','offset','offset2','first_scale','second_scale','r2','objective'))
    if mask_checks:
        branches['status_exact']=bool(np.array_equal(got['status'],ref['status'])) if 'status' in ref else None
        branches['initial_status_exact']=bool(np.array_equal(got['initial_status'],ref['initial_status'])) if 'initial_status' in ref else None
    return dict(pass_all=all(x['pass_gate'] for x in checks.values()) and
                     all(v is not False for k,v in branches.items() if k.endswith('_exact')) and
                     branches['all_numerical_fields_finite'],
                checks=checks,branches=branches)


def compare_moments_only(got, ref, model, precision):
    gates=PRECISION_GATES[precision]; checks={}
    keys=('ms','mu') if model=='deterministic' else ('ms','mu','mss','mus')
    for key in keys:
        checks[key]=field_gate(got[key],ref[key],gates['moment_atol'],gates['moment_rtol'])
    return dict(pass_all=all(v['pass_gate'] for v in checks.values()),coverage='independent_moments_only_for_analytic_branch_fixture',checks=checks)


def r2_diagnostic(got, ref, stored, fit_mask, fit_offset, trimming, failed):
    """Explain an oracle R² miss without changing its acceptance gate."""
    if failed.get('pass_gate', True):
        return None
    ms64,mu64,_,_=oracle_moments(*stored,np.float64)
    rows=np.flatnonzero(fit_mask)
    native_gamma=np.asarray(got['initial_gamma'],dtype=np.float64)
    native_offset=np.asarray(got['initial_offset'],dtype=np.float64)
    source_gamma=np.asarray(ref['initial_gamma'],dtype=np.float64)
    source_offset=np.asarray(ref['initial_offset'],dtype=np.float64)

    def r2_with(gamma,offset):
        residual=mu64-gamma[None,:]*ms64-offset[None,:]
        total=mu64-np.mean(mu64,axis=0,keepdims=True)
        den=np.sum(total*total,axis=0,dtype=np.float64)
        num=np.sum(residual*residual,axis=0,dtype=np.float64)
        with np.errstate(all='ignore'):
            values=1.-num/den
        values[np.isnan(values)]=0.
        return values

    native_precise=r2_with(native_gamma,native_offset)
    source_precise=r2_with(source_gamma,source_offset)
    exact_gamma=np.zeros(ms64.shape[1],dtype=np.float64)
    exact_offset=np.zeros_like(exact_gamma)
    exact_fit_r2=np.zeros_like(exact_gamma)
    exact_rows=np.zeros(ms64.shape[1],dtype=np.int64)
    for j in range(ms64.shape[1]):
        if not rows.size:
            continue
        x=np.asarray(ms64[rows,j],dtype=np.float64)
        y=np.asarray(mu64[rows,j],dtype=np.float64)
        if trimming=='none':
            keep=np.ones(rows.size,dtype=bool)
        else:
            score=x/max(float(np.max(x)),1e-3)+y/max(float(np.max(y)),1e-3)
            if trimming=='tails':
                qlo,qhi=np.percentile(score,(5,95),method='linear')
                keep=((score<=qlo)|(score>=qhi)) if fit_offset else score>=qhi
            else:
                qhi=np.percentile(score,95,method='linear')
                keep=score>=qhi
        design=np.column_stack((x[keep],np.ones(np.count_nonzero(keep)))) if fit_offset else x[keep,None]
        if not design.size:
            continue
        beta,*_=np.linalg.lstsq(design,y[keep],rcond=None)
        exact_gamma[j]=beta[0]
        if fit_offset:
            exact_offset[j]=beta[1]
        exact_rows[j]=int(np.count_nonzero(keep))
    exact_fit_r2=r2_with(exact_gamma,exact_offset)
    errors=np.abs(np.asarray(got['r2'],dtype=np.float64)-np.asarray(ref['r2'],dtype=np.float64))
    worst=np.argsort(errors)[::-1]
    genes=[]
    for j in worst:
        if errors[j] == 0:
            break
        genes.append(dict(gene=int(j),native_r2=float(got['r2'][j]),source_r2=float(ref['r2'][j]),
            absolute_delta=float(errors[j]),native_r2_fp64_exact_input=float(native_precise[j]),
            source_r2_fp64_exact_input=float(source_precise[j]),
            native_r2_evaluation_error=float(got['r2'][j]-native_precise[j]),
            source_r2_evaluation_error=float(ref['r2'][j]-source_precise[j]),
            native_initial_gamma=float(native_gamma[j]),native_initial_offset=float(native_offset[j]),
            source_initial_gamma=float(source_gamma[j]),source_initial_offset=float(source_offset[j]),
            exact_input_fp64_ols_gamma=float(exact_gamma[j]),exact_input_fp64_ols_offset=float(exact_offset[j]),
            exact_input_fp64_ols_r2=float(exact_fit_r2[j]),exact_input_fit_rows=int(exact_rows[j])))
        if len(genes)==5:
            break
    return dict(method='exact_stored_values_cast_to_fp64; independent SciPy moments; direct NumPy LAPACK OLS',
        interpretation='R² evaluation and fit-parameter effects are decomposed; nominal oracle gate remains unchanged',
        highest_delta_genes=genes)


def branch_contract(spec, outputs, model, baseline):
    if spec['fixture']!='analytic_degenerate':
        return None
    result=outputs[baseline]
    names={
        'deterministic':{0:'ok',1:'insufficient_data',2:'no_signal',3:'singular',4:'constant_response'},
        'stochastic':{0:'ok',1:'not_selected',2:'invalid_initialization',3:'insufficient_data',4:'zero_scale',5:'singular',6:'nonfinite'}
    }
    initial_names={0:'ok',1:'insufficient_data',2:'no_signal',3:'singular',4:'constant_response'}
    per_gene={str(j):dict(initial_status=initial_names.get(int(result['initial_status'][j]),str(int(result['initial_status'][j]))),
                         final_status=names[model].get(int(result['status'][j]),str(int(result['status'][j]))),
                         selected=bool(result['selected'][j])) for j in range(result['selected'].size)}
    expected=dict(
        deterministic='gene 1 and 2 have no signal; gene 3 has constant spliced input under the intercept fit; gene 4 has constant response; gene 5 is disabled by the gene mask',
        stochastic='deterministic initialization branches remain visible; selected genes report joint-fit or fallback status')
    if spec['mask']=='single_fit_row':
        expected['deterministic']='gene 1 and 2 have no signal; one fitting row makes the remaining intercept fits insufficient or singular'
        expected['stochastic']='single-row fit population cannot produce a successful joint refinement'
    passed=True
    if spec['mask']=='all' and model=='deterministic':
        passed=(result['initial_status'][1]==2 and result['initial_status'][2]==2 and
                result['initial_status'][3]==3 and result['initial_status'][4]==4 and
                result['initial_status'][0]==0 and result['initial_status'][5]==0 and not result['selected'][5])
    if spec['mask']=='single_fit_row' and model=='stochastic':
        passed=not bool(np.any(result['status']==0))
    return dict(pass_contract=bool(passed),expected=expected[model],per_gene=per_gene,
                status_counts={str(int(k)):int(v) for k,v in zip(*np.unique(result['status'],return_counts=True))},
                initial_status_counts={str(int(k)):int(v) for k,v in zip(*np.unique(result['initial_status'],return_counts=True))})


def read_condition(result, mask, fit_offset, fit_offset2, trimming, model):
    a,b=conditions(result['ms'],result['mu'],mask,fit_offset,fit_offset2,trimming,model,result)
    return {'initial':a,'joint':b}


def synthetic_inputs():
    n,g=12,6
    x=np.arange(1,n+1,dtype=np.float32)
    s=np.empty((n,g),np.float32);u=np.empty_like(s)
    s[:,0]=x;u[:,0]=2+.8*x+np.array([0,.1,-.1,.05,-.04,.06,-.08,.03,.02,-.05,.07,-.03],np.float32)
    s[:,1]=0;u[:,1]=0
    s[:,2]=x;u[:,2]=0
    s[:,3]=4;u[:,3]=.5*x+1
    s[:,4]=x;u[:,4]=3
    s[:,5]=.5*x+1;u[:,5]=1.5*x+.25
    w=np.eye(n,dtype=np.float32)
    fit=np.ones(n,np.uint8);genes=np.ones(g,np.uint8);genes[5]=0
    return (s,u,w),fit,genes


def profile_quantization(canonical, stored):
    labels=('spliced','unspliced','weights');out={}
    for name,original,actual in zip(labels,canonical,stored):
        restored=np.asarray(actual,dtype=np.float64)
        base=np.asarray(original,dtype=np.float64)
        diff=np.abs(restored-base)
        out[name]=dict(storage_dtype=str(actual.dtype),changed_values=int(np.count_nonzero(diff)),
                       values=int(diff.size),max_absolute_error=float(np.max(diff,initial=0.)),
                       rms_error=float(np.sqrt(np.mean(diff*diff))) if diff.size else 0.)
    return out


def make_specs(name, hvg, output_dir):
    n,g=100,int(hvg.size)
    all_fit=np.ones(n,np.uint8); all_genes=np.ones(g,np.uint8)
    fit=np.ones(n,np.uint8);fit[::3]=0
    variants=[]
    def add(profile,compute,mask_name,fitmask,genemask,config):
        variants.append(dict(fixture=name,profile=profile,compute=compute,mask=mask_name,
            fit_mask=fitmask.copy(),gene_mask=genemask.copy(),**config))
    base={'fit_offset':False,'fit_offset2':False,'trimming':'tails','config':'default'}
    for compute in ('f32','f64'):
        add('fp32',compute,'all',all_fit,all_genes,base)
    add('fp64','f64','all',all_fit,all_genes,base)
    add('fp16','f32','all',all_fit,all_genes,base)
    add('mixed_s16_u32_w64','f32','all',all_fit,all_genes,base)
    add('mixed_s64_u16_w32','f32','all',all_fit,all_genes,base)
    for compute in ('f32','f64'):
        add('fp32',compute,'all',all_fit,all_genes,
            {'fit_offset':True,'fit_offset2':True,'trimming':'tails','config':'joint_intercepts'})
        add('fp32',compute,'fit_subset_hvg',fit,hvg.astype(np.uint8),base)
    if name=='dentategyrus':
        for mask_name,fitmask,genemask in (('all',all_fit,all_genes),('fit_subset_hvg',fit,hvg.astype(np.uint8))):
            for fit_second in (False,True):
                add('fp32','f32',mask_name,fitmask,genemask,
                    {'fit_offset':True,'fit_offset2':fit_second,'trimming':'none',
                     'config':f'm2_accepted_source_exception_offset2_{int(fit_second)}'})
    arrays,deg_fit,deg_genes=synthetic_inputs()
    for compute in ('f32','f64'):
        for mask_name,fitmask in (('all',deg_fit),('single_fit_row',None)):
            # The analytic branch fixture has six genes and twelve observations.
            if mask_name=='single_fit_row':
                fitmask=np.zeros(12,np.uint8);fitmask[0]=1
            else: fitmask=np.asarray(fitmask,dtype=np.uint8)
            variants.append(dict(fixture='analytic_degenerate',profile='fp32',compute=compute,
                mask=mask_name,fit_mask=fitmask,gene_mask=deg_genes.copy(),fit_offset=True,
                fit_offset2=True,trimming='tails',config='explicit_branches',
                arrays=arrays,fixture_sha256=sha256(b''.join(np.ascontiguousarray(x).tobytes() for x in arrays))))
    # Attach fixed h5ad arrays later from the fixture cache.
    return variants


def source_hashes():
    files=[p for p in sorted((ROOT/'src').rglob('*')) if p.is_file()]
    return {str(p.relative_to(ROOT)):sha256(p.read_bytes()) for p in files}


def run_driver(driver: Path, input_path: Path, output_path: Path, spec, method, device, timeout):
    cmd=[str(driver),str(input_path),str(output_path),method,spec['compute'],
         str(int(spec['fit_offset'])),str(int(spec['fit_offset2'])),spec['trimming'],'.01','.01']
    if 'cuda' in driver.name.lower() or 'cuda' in str(driver).lower():
        cmd.append(str(device))
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    proc=subprocess.run(cmd,check=False,capture_output=True,text=True,timeout=timeout,env=env)
    if proc.returncode:
        raise RuntimeError(f"driver failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return cmd


def stored_for_spec(spec, fixture_cache):
    if 'arrays' in spec:
        original=tuple(np.asarray(x,dtype=np.float32) for x in spec['arrays'])
        hvg=spec['gene_mask'].astype(bool)
        path=None
    else:
        path,s,u,w,hvg=fixture_cache[spec['fixture']]
        original=(s,u,w)
    stored=quantize_inputs(*original,spec['profile'])
    return path,original,stored,hvg


def run_qualification(args):
    ns=qs.source_namespace()
    run_tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}'
    run_dir=args.output.parent/'precision'/run_tag
    raw_dir=run_dir/'raw';input_dir=run_dir/'inputs'
    raw_dir.mkdir(parents=True,exist_ok=False);input_dir.mkdir(parents=True,exist_ok=False)
    fixture_cache={name:fixture_arrays(name) for name in ('pancreas','dentategyrus')}
    specs=[]
    for name,(_,s,u,w,hvg) in fixture_cache.items():
        specs.extend(make_specs(name,hvg,input_dir))
    if args.max_specs:
        specs=specs[:args.max_specs]
    drivers={name:Path(path).resolve() for name,path in (('scalar',args.scalar),('avx2',args.avx2),('cuda',args.cuda)) if path}
    if not drivers: raise ValueError('provide at least one backend driver')
    for name,path in drivers.items():
        if not path.is_file(): raise FileNotFoundError(path)
    backend_ref='scalar' if 'scalar' in drivers else ('avx2' if 'avx2' in drivers else next(iter(drivers)))
    report=dict(
        status='incomplete' if len(drivers)<3 else 'running',
        created_utc=datetime.now(timezone.utc).isoformat(),run_id=run_tag,
        acceptance=dict(scope='same serialized input bytes, masks, method and options across scalar, compiled CPU AVX2 and CUDA',
            per_gene_scale='max(1,max(abs(reference)))',condition_limit=1e3,
            branch_policy='selection and status outputs are reported exactly; degenerate analytic branch cases are explicit',
            gates=PRECISION_GATES,
            historical_m2_exception='The previously accepted Milestone 2 source FP32 initialization exception remains recorded in its original report. This harness does not edit or reclassify that report.'),
        provenance=dict(source_commit=qs.SOURCE_COMMIT,source_hashes=qs.SOURCE_HASHES,
            native_source_sha256=source_hashes(),drivers={k:dict(path=str(v),sha256=sha256(v.read_bytes())) for k,v in drivers.items()},
            qualification_driver_sha256=sha256((ROOT/'tests/precision_interop_driver.cpp').read_bytes()),
            qualification_script_sha256=sha256(Path(__file__).read_bytes()),
            fixture_paths={n:str(v[0]) for n,v in fixture_cache.items()},
            fixture_sha256={n:sha256(v[0].read_bytes()) for n,v in fixture_cache.items()},
            python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version('scipy'),
            timeout_seconds=args.timeout,raw_artifacts=str(run_dir)),
        backend_reference=backend_ref,cases=[],storage_quantization=[],
        missing_backends=[k for k in ('scalar','avx2','cuda') if k not in drivers])
    by_case={}
    for index,spec in enumerate(specs):
        path,original,stored,hvg=stored_for_spec(spec,fixture_cache)
        n,g=stored[0].shape
        key='_'.join((spec['fixture'],spec['profile'],spec['compute'],spec['mask'],spec['config']))
        input_path=input_dir/f'{key}.bin'
        input_sha=write_input(input_path,stored,spec['fit_mask'],spec['gene_mask'])
        compute=np.float32 if spec['compute']=='f32' else np.float64
        oracle_cache={}
        for model in ('deterministic','stochastic'):
            if spec['fixture']=='analytic_degenerate':
                ms,mu,mss,mus=oracle_moments(*stored,compute)
                oracle=dict(ms=ms,mu=mu,mss=mss,mus=mus)
            else:
                oracle=independent_oracle(ns,stored,spec['fit_mask'].astype(bool),spec['gene_mask'].astype(bool),compute,model,
                    spec['fit_offset'],spec['fit_offset2'],spec['trimming'])
            oracle_cache[model]=oracle
            outputs={}
            commands={}
            for backend,driver in drivers.items():
                out_path=raw_dir/f'{key}.{model}.{backend}.bin'
                cmd=run_driver(driver,input_path,out_path,spec,model,args.device,args.timeout)
                commands[backend]=cmd
                got=read_output(out_path,n,g)
                if got['precision']!=spec['compute'] or got['method']!=model:
                    raise RuntimeError(f'output tag mismatch for {key}/{model}/{backend}')
                expected_backend={'scalar':0,'avx2':1,'cuda':2}[backend]
                if got['backend_code']!=expected_backend:
                    raise RuntimeError(f'{backend} driver reports backend code {got["backend_code"]}; expected {expected_backend}')
                outputs[backend]=got
            ref_backend=outputs[backend_ref]
            # Per-backend source comparison is additional to the same-input backend comparisons.
            if spec['fixture']=='analytic_degenerate':
                oracle_checks=compare_moments_only(ref_backend,oracle,model,spec['compute'])
            else:
                condition=read_condition(oracle,spec['fit_mask'].astype(bool),spec['fit_offset'],spec['fit_offset2'],spec['trimming'],model)
                oracle_checks=compare_results(ref_backend,oracle,model,spec['compute'],condition)
            r2_diag=None
            if 'r2' in oracle_checks.get('checks',{}):
                r2_diag=r2_diagnostic(ref_backend,oracle,stored,spec['fit_mask'].astype(bool),
                    spec['fit_offset'],spec['trimming'],oracle_checks['checks']['r2'])
            pairwise={}
            for backend,got in outputs.items():
                if backend==backend_ref: continue
                pairwise[backend]=compare_results(got,ref_backend,model,spec['compute'],
                    read_condition(ref_backend,spec['fit_mask'].astype(bool),spec['fit_offset'],spec['fit_offset2'],spec['trimming'],model))
                pairwise[backend]['boundary_rank_diagnostics']=boundary_summary(ref_backend,got,spec['fit_mask'].astype(bool),spec['fit_offset'],spec['trimming'])
            record=dict(case_id=f'{key}.{model}',fixture=spec['fixture'],fixture_sha256=sha256(path.read_bytes()) if path else spec.get('fixture_sha256'),
                stored_profile=spec['profile'],stored_dtypes=[str(x.dtype) for x in stored],compute_precision=spec['compute'],
                input_path=str(input_path),input_sha256=input_sha,fit_mask=spec['mask'],model=model,
                fit_offset=spec['fit_offset'],fit_second_intercept=spec['fit_offset2'],trimming=spec['trimming'],
                comparison=dict(reference_backend=backend_ref,independent_oracle=oracle_checks,pairwise=pairwise),
                same_input_backend_parity_pass=all(x['pass_all'] for x in pairwise.values()),
                independent_source_oracle_pass=bool(oracle_checks['pass_all']),
                independent_oracle_r2_diagnostic=r2_diag,
                backend_commands=commands,
                backend_artifacts={b:dict(path=str(raw_dir/f'{key}.{model}.{b}.bin'),
                    sha256=sha256((raw_dir/f'{key}.{model}.{b}.bin').read_bytes())) for b in outputs},
                boundary_quantiles=boundary_summary(oracle,ref_backend,spec['fit_mask'].astype(bool),spec['fit_offset'],spec['trimming']),
                backend_status={b:{'selected':int(np.count_nonzero(r['selected'])),
                    'status_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(r['status'],return_counts=True))},
                    'initial_status_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(r['initial_status'],return_counts=True))},
                    'all_outputs_finite':all(bool(np.isfinite(r[f]).all()) for f in ('ms','mu','velocity','gamma','r2'))}
                    for b,r in outputs.items()},
                quantization_vs_original=profile_quantization(original,stored),
                explicit_branch_contract=branch_contract(spec,outputs,model,backend_ref))
            report['cases'].append(record)
            by_case[(spec['fixture'],spec['profile'],spec['compute'],spec['mask'],spec['config'],model)]=outputs
            if spec['profile']!='fp32' and spec['mask']=='all' and spec['config']=='default' and spec['compute']=='f32':
                base=by_case.get((spec['fixture'],'fp32','f32','all','default',model),{}).get(backend_ref)
                quant=outputs[backend_ref]
                if base is not None:
                    report['storage_quantization'].append(dict(fixture=spec['fixture'],model=model,
                        canonical_profile='fp32',quantized_profile=spec['profile'],compute_precision='f32',
                        interpretation='storage quantization effect; excluded from backend mismatch gates',
                        changed_selected_genes=int(np.count_nonzero(base['selected']!=quant['selected'])),
                        velocity_max_abs_delta=float(np.max(np.abs(base['velocity']-quant['velocity']))),
                        gamma_max_abs_delta=float(np.max(np.abs(base['gamma']-quant['gamma']))),
                        profile_input_quantization=profile_quantization(original,stored)))
    # Compare FP32 and FP64 arithmetic on identical FP32 stored bytes. These
    # precision-pair gates use the FP32 allowance and remain separate from
    # quantization comparisons between different serialized storage types.
    report['precision_comparisons']=[]
    spec_map={(s['fixture'],s['profile'],s['compute'],s['mask'],s['config']):s for s in specs}
    for low_spec in specs:
        if low_spec['profile']!='fp32' or low_spec['compute']!='f32': continue
        high_spec=spec_map.get((low_spec['fixture'],'fp32','f64',low_spec['mask'],low_spec['config']))
        if high_spec is None: continue
        for model in ('deterministic','stochastic'):
            low=by_case.get((low_spec['fixture'],'fp32','f32',low_spec['mask'],low_spec['config'],model))
            high=by_case.get((low_spec['fixture'],'fp32','f64',low_spec['mask'],low_spec['config'],model))
            if low is None or high is None: continue
            by_backend={}
            for backend in low.keys() & high.keys():
                condition=read_condition(high[backend],high_spec['fit_mask'].astype(bool),
                    high_spec['fit_offset'],high_spec['fit_offset2'],high_spec['trimming'],model)
                by_backend[backend]=compare_results(low[backend],high[backend],model,'f32',condition)
            report['precision_comparisons'].append(dict(fixture=low_spec['fixture'],
                stored_profile='fp32',fit_mask=low_spec['mask'],config=low_spec['config'],model=model,
                input_bytes_identical=True,compute_pair='f32_vs_f64',gate_precision='f32',backends=by_backend,
                pass_all=all(x['pass_all'] for x in by_backend.values())))
    report['driver_runs']=len(report['cases'])*len(drivers)
    report['backend_complete']=all(k in drivers for k in ('scalar','avx2','cuda'))
    report['pass_all_backend_gates']=all(all(v['pass_all'] for v in c['comparison']['pairwise'].values()) for c in report['cases'])
    report['backend_parity_status']=('pass' if report['backend_complete'] and report['pass_all_backend_gates'] else
        'fail' if any(not c['same_input_backend_parity_pass'] for c in report['cases']) else 'incomplete')
    report['pass_all_oracle_gates']=all(c['comparison']['independent_oracle']['pass_all'] for c in report['cases'])
    report['pass_all_precision_gates']=all(x['pass_all'] for x in report['precision_comparisons'])
    branch_checks=[c['explicit_branch_contract']['pass_contract'] for c in report['cases'] if c['explicit_branch_contract'] is not None]
    report['pass_all_branch_contracts']=all(branch_checks)
    report['pass_all']=report['backend_complete'] and report['pass_all_backend_gates'] and report['pass_all_oracle_gates'] and report['pass_all_precision_gates'] and report['pass_all_branch_contracts']
    report['status']='pass' if report['pass_all'] else 'fail' if report['backend_complete'] else 'incomplete'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),driver_runs=report['driver_runs'],
        pass_all_backend_gates=report['pass_all_backend_gates'],pass_all_oracle_gates=report['pass_all_oracle_gates'],
        missing_backends=report['missing_backends'],output=str(args.output),raw_artifacts=str(run_dir)),indent=2))
    return 0 if report['pass_all'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scalar',type=Path)
    p.add_argument('--avx2',type=Path)
    p.add_argument('--cuda',type=Path)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--timeout',type=float,default=30.0)
    p.add_argument('--max-specs',type=int,default=0,help='limit input configurations for a bounded smoke run')
    p.add_argument('--output',type=Path,default=ROOT/'tests/artifacts/precision-qualification.json')
    args=p.parse_args()
    if args.timeout<=0: p.error('--timeout must be positive')
    raise SystemExit(run_qualification(args))


if __name__=='__main__':
    main()
