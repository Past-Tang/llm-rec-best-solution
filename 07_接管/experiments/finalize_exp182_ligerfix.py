#!/usr/bin/env python3
"""Verify the locked EXP-182 Liger-fix preregistration and finalize its audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import audit_recovery_run, run_training

REPO=Path(__file__).resolve().parents[2]; ROOT=REPO/'07_接管/experiments'
DESIGN=ROOT/'preregistrations/exp182_headfix_reproduction_liger_frozen_head_fix.json'
CONFIG=ROOT/'configs/exp182_headfix_reproduction_step10_fp16_xformers_ligerfix.yaml'
PATCH=ROOT/'liger_frozen_head_patch.py'; SITE=ROOT/'frozen_head_liger_site/sitecustomize.py'; LAUNCHER=ROOT/'run_frozen_head_training.py'
DATA=ROOT/'data/r0_cbalanced_replay_v1.jsonl'; ANCESTOR=ROOT/'outputs/onereason_0.8b_takeover_r0_aligned_replay_v8t_freeze8_sg16'
TARGET=ROOT/'outputs/onereason_0.8b_exp174_exp173_negative_half_merged_bf16'; EXPERT=ROOT/'outputs/onereason_0.8b_exp182_headfix_reproduction_step10_fp16_xformers_ligerfix'
RUN=ROOT/'runs/exp182_headfix_reproduction_step10_fp16_xformers_ligerfix_v1'; NAME='exp182_exp174_headfix_repro_ligerfix_negative_half_abc_v1'
OUTPUT=ROOT/'selective_task_vector_models'/NAME; COMPOSITION=ROOT/'manifests'/f'{NAME}.selective_task_vector_manifest.json'
def req(v,m):
    if not v: raise RuntimeError(m)
def rec(p): return run_training.file_record(p)
def verify():
    d=json.loads(DESIGN.read_text()); req(d['experiment']=='EXP-182' and d['status']=='LOCKED_BEFORE_TRAINING_DERIVATION_UPLOAD_OR_FORMAL','design drift')
    for k,p in [('config',CONFIG),('launcher',LAUNCHER),('patch',PATCH),('sitecustomize',SITE),('dataset',DATA)]: req(d['training'][k]==rec(p),f'{k} drift')
    req(run_training.sha256_file(ANCESTOR/'model.safetensors')==d['training']['initializer_sha256'],'initializer drift')
    req(run_training.sha256_file(TARGET/'model.safetensors')==d['derivation']['target_sha256'],'target drift')
    return d
def finalize():
    d=verify(); c=json.loads((RUN/'completion_manifest.json').read_text()); req(c['verdict']=='PASS' and c['output']['global_step']==10,'completion drift')
    m=json.loads(COMPOSITION.read_text()); req(m['groups']==['a','b','c'] and m['alpha']=='-0.5','composition drift')
    finite=audit_recovery_run.scan_safetensors(OUTPUT/'model.safetensors'); req(finite['all_finite'] and finite['tensor_count']==311,'finite scan failed')
    a={'schema':'llm-rec.takeover.training-run-audit','schema_version':1,'experiment':'EXP-182','run_id':RUN.name,'verdict':'PASS','lineage_completeness':'FULL','audit_kind':'frozen_head_liger_corrected_training_plus_selective_task_vector','inputs':{'preregistration':rec(DESIGN),'completion':rec(RUN/'completion_manifest.json'),'composition':rec(COMPOSITION),'expert':rec(EXPERT/'model.safetensors'),'target':rec(TARGET/'model.safetensors')},'finite_scan':finite,'output':{'directory':str(OUTPUT),'artifacts':{p.name:rec(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}},'formal_policy':d['formal_policy']}
    p=RUN/'audit_manifest.json'; req(not p.exists(),'audit exists'); run_training.atomic_write(p,run_training.canonical_json_bytes(a)); run_training.atomic_write(p.with_suffix('.json.sha256'),f'{run_training.sha256_file(p)}  {p.name}\n'.encode()); return a
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--finalize',action='store_true'); args=ap.parse_args()
    try: print(json.dumps({'verdict':'PASS','mode':'finalize' if args.finalize else 'verify','result':finalize() if args.finalize else verify()},ensure_ascii=False))
    except Exception as e: print(f'FAIL: {e}'); raise SystemExit(1)
