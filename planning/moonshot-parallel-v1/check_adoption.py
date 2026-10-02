#!/usr/bin/env python3
"""Check immutable seeds and the agreed experimental interfaces; no GPU launch."""
import argparse,hashlib,json,pathlib,sys

def check(root):
    interface=root/'experiments/moonshot-parallel-v1/interfaces/contract.json'
    contract=json.loads(interface.read_text())
    assert contract['schema_version']==1 and contract['record_kind']=='moonshot_shared_prototype_contract'
    assert contract['native_ABI'] is False
    seeds=root/'experiments/moonshot-parallel-v1/seeds'
    original=root/contract['seeds']['source']
    files=contract['seeds']['files']
    actual={p.relative_to(seeds).as_posix() for p in seeds.rglob('*') if p.is_file()}
    assert set(files)==actual,'seed file set changed'
    manifest=json.loads((root/'planning/moonshot-v1/MANIFEST.json').read_text())['files']
    for name,digest in files.items():
        rel=pathlib.PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        seed=seeds/rel
        assert not seed.is_symlink() and not (seed.stat().st_mode & 0o222),'seed must remain immutable'
        assert hashlib.sha256(seed.read_bytes()).hexdigest()==digest==manifest['payload/cellerator/'+name]
        assert seed.read_bytes()==(original/rel).read_bytes()
    state=contract['state']
    assert state['logical_key']==['actor_id','local_slot','slot_incarnation']
    assert state['generations']==['structure_epoch','value_generation','activity_generation','parameter_generation']
    assert set(state['separate_masks'])=={'structural_support','trainable_capacity','forward_activity','response_support'}
    assert 'one recorded' in state['snapshot'] and 'old readers/tapes' in state['publication']
    assert set(contract['derivatives']['receipt_fields'])=={'forward','input_vjp','parameter_vjp','jvp','second_order'}
    assert contract['derivatives']['default_status']=='not_implemented until lane evidence'
    assert 'does not mutate' in contract['derivatives']['saved_primal']
    assert 'parameter-gradient support' in contract['derivatives']['trainable_zero']
    assert 'one canonical owner' in contract['derivatives']['replicas']
    assert 'STE surrogate' in contract['precision']['patch16']['derivative_policy']
    assert set(contract['admission'])=={'dtype','shape_and_extents','device_and_stream','alignment','index_bounds','output_aliasing','capacity','generation_match','deterministic_padding'}
    assert 'one declared owner' in contract['outputs']['ownership']
    assert len(set(contract['lanes'].values()))==5 and all(x.startswith('experiments/moonshot-parallel-v1/') for x in contract['lanes'].values())
    assert contract['preservation']['unchanged_existing_tasks'] is True
    assert contract['claims']['performance']=='not_measured'
    return len(files)
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'));args=ap.parse_args()
    try:print(f'Adoption passed: {check(args.root)} immutable archive-matched seeds; shared identity, precision, derivative, admission and ownership contracts present. No CUDA/science qualification.')
    except Exception as exc:print(f'Adoption failed: {exc}',file=sys.stderr);sys.exit(1)
