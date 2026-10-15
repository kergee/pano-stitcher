"""Repeatable controlled matrix, original hash-list real photos, isolated RSS measurements.

python tools/verify_changes.py --before output/verification-20261015/before --real-manifest PATH
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from tools.make_testset import generate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'output/verification-20261015')
    ap.add_argument('--before',type=Path)
    ap.add_argument('--real-manifest',type=Path)
    ap.add_argument('--repeats',type=int,default=3)
    args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    scenarios=['normal','lowoverlap','lowtex','dark','extreme','repeat','mixed']
    manifests={}
    for name in scenarios:
        folder=out/'bench'/name
        if not (folder/'inputs.json').exists():generate(name,folder)
        manifests[name]=(folder/'inputs.json').resolve()
    if args.real_manifest:manifests['real7']=args.real_manifest.resolve()
    variants={'default':{},'no-local':{'local_warp':False},'staged':{'ba_mode':'staged'},
              'ordered':{'pair_mode':'ordered'},'fast':{'quality':'fast'},'small':{'max_output_side':480}}
    tasks=[]
    for name in manifests:
        repeat=args.repeats if name in ('normal','lowoverlap','real7') else 1
        for k in range(repeat):
            tasks.append((name,'after','default',k,ROOT,'pro',{}))
            if args.before:tasks.append((name,'before','default',k,args.before.resolve(),'pro',{}))
        tasks.append((name,'after','baseline',0,ROOT,'baseline',{}))
        if name in ('normal','lowoverlap','real7','repeat','mixed'):
            for variant,options in variants.items():
                if variant=='default':continue
                tasks.append((name,'after',variant,0,ROOT,'pro',options))
    rows=[]
    for name,version,variant,k,source,engine,options in tasks:
        target=out/'images'/f'{name}-{version}-{variant}-{k}.jpg'
        cmd=[sys.executable,str(ROOT/'tools/verification_worker.py'),'--source',str(source),'--manifest',str(manifests[name]),
             '--output',str(target),'--engine',engine,'--options',json.dumps(options)]
        proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'},timeout=120)
        try:row=json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError,IndexError):raise RuntimeError(f'Worker failed: {proc.stdout}\n{proc.stderr}')
        row.update(case=name,version=version,variant=variant,repeat=k,exit_code=proc.returncode)
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,version,variant,k,'generated='+str(row['generated']),f"{row['seconds']:.3f}s",f"peak={row.get('peak_working_set_bytes',0)/1024**2:.1f}MiB",flush=True)
    summary=[]
    keys=sorted(set((r['case'],r['version'],r['variant']) for r in rows))
    for name,version,variant in keys:
        group=[r for r in rows if (r['case'],r['version'],r['variant'])==(name,version,variant)]
        summary.append({'case':name,'version':version,'variant':variant,'runs':len(group),
                        'generated':all(r['generated'] for r in group),'geometry_pass':None if all(r.get('geometry_pass') is None for r in group) else all(r.get('geometry_pass',False) for r in group),
                        'seconds_median':statistics.median(r['seconds'] for r in group),
                        'peak_mib_median':statistics.median(r.get('peak_working_set_bytes',0)/1024**2 for r in group),
                        'focal':group[0].get('focal'),'used':group[0].get('used'),'size':group[0].get('size')})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Expected difficult SIFT rejection is evidence, not a harness failure.
    required=[r for r in rows if r['version']=='after' and r['variant']=='default' and r['case'] in ('normal','lowoverlap','mixed','real7')]
    return int(any(not r['generated'] or (r['case']!='real7' and not r.get('geometry_pass')) for r in required))


if __name__=='__main__':raise SystemExit(main())
