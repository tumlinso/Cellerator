#!/usr/bin/env python3
"""Render static result figures; never execute a benchmark or invent pending data."""
from __future__ import annotations
import argparse
import textwrap
from pathlib import Path
from common import read_json, validate_result

def draw(record: dict, out: Path):
    validate_result(record)
    if record.get('chart')=='table':return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import StrMethodFormatter
    out.mkdir(parents=True,exist_ok=True)
    rows=record['rows'];series=record['series']
    interval=record.get('chart')=='interval'
    fig,ax=plt.subplots(figsize=(10.5,6.7) if interval else (9.5,5.6))
    title='\n'.join(textwrap.wrap(record['title'],64))
    fig.suptitle(title,fontsize=15,y=.97)
    two_lines='\n' in title
    fig.text(.5,.85 if two_lines else .885,record.get('figure_context',''),ha='center',fontsize=10)
    unit=record['metric']['unit'];scale=1.0
    if unit=='ns':unit='µs';scale=.001
    elif unit=='ms':unit='µs';scale=1000
    elif unit=='us':unit='µs'
    if not rows:
        ax.set_axis_off()
        ax.text(.5,.55,'Measurement pending',ha='center',va='center',fontsize=18,transform=ax.transAxes)
        ax.text(.5,.39,'Run the actual library fixture before filling this chart.',ha='center',va='center',fontsize=11,transform=ax.transAxes)
    elif interval:
        pos=np.arange(len(rows));offset=.15
        for i,s in enumerate(series):
            ys=pos+(i-(len(series)-1)/2)*2*offset
            values=np.array([r['values'][s['key']] for r in rows])*scale
            bounds=[r.get('bounds',{}).get(s['key'],[r['values'][s['key']]]*2) for r in rows]
            err=np.array([[v-b[0]*scale for v,b in zip(values,bounds)],[b[1]*scale-v for v,b in zip(values,bounds)]])
            ax.errorbar(values,ys,xerr=err,fmt=['o','s','^','D'][i%4],capsize=3,label=s['label'],markersize=5)
        ax.set_yticks(pos,[r['case'].replace(' · ','\n',1) for r in rows]);ax.invert_yaxis()
        ax.set_xlabel(f'Lifecycle time ({unit}); points are medians, whiskers are min–max')
        ax.set_xlim(left=0);ax.legend(loc='upper left',bbox_to_anchor=(0,1.11),ncol=2,frameon=False,fontsize=10)
        fig.subplots_adjust(left=.36,right=.98,top=.71 if two_lines else .77,bottom=.23)
    else:
        pos=np.arange(len(rows));width=.78/len(series)
        for i,s in enumerate(series):
            y=[r['values'][s['key']]*scale for r in rows]
            bars=ax.bar(pos+(i-(len(series)-1)/2)*width,y,width,label=s['label'],hatch=['','//','..','xx'][i%4])
            ax.bar_label(bars,labels=[f'{v:,.4g}' for v in y],padding=3,fontsize=9)
        ax.set_xticks(pos,[r['case'] for r in rows]);ax.set_ylim(bottom=0)
        lo,hi=ax.get_ylim();ax.set_ylim(0,hi*1.20)
        ax.set_ylabel(f"Reported time ({unit})" if unit in ['µs','ms','s'] else f"{record['metric']['statistic'].capitalize()} value ({unit})")
        ax.legend(loc='upper right',frameon=False,fontsize=9)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.4g}'))
        fig.subplots_adjust(left=.12,right=.98,top=.75 if two_lines else .80,bottom=.23)
    state={'recorded_summary':'Recorded source summary · not rerun for this package',
           'pending':'No product measurement exists',
           'verified_historical':'Verified historical measurement · not a new run',
           'fresh_measurement':'Measured run · exact source/method in the linked record'}[record['evidence_state']]
    note=state+'\n'+record['metric']['timing_scope']
    fig.text(.04,.10,'\n'.join(textwrap.wrap(note.split('\n')[0],115)),fontsize=9,va='top')
    fig.text(.04,.065,'\n'.join(textwrap.wrap(note.split('\n')[1],115)),fontsize=8,va='top')
    paths=[]
    for ext in ['png','svg']:
        path=out/f"{record['id']}.{ext}"
        fig.savefig(path,dpi=160,bbox_inches='tight');paths.append(path)
    plt.close(fig)
    return paths

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('record',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();draw(read_json(a.record),a.out)
if __name__=='__main__':main()
