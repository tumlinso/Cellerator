#!/usr/bin/env python3
"""Verify a CUDA controller receipt against the installed read-only host owner API."""
import argparse
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def process_start(pid):
    # Linux comm may contain spaces and parentheses. Fields after the final ')'
    # begin at field 3; starttime is field 22.
    raw = Path('/proc') / str(pid) / 'stat'
    return raw.read_text().rsplit(')', 1)[1].split()[19]


def verify(receipt, owner, expected_project, expected_gpus, read_start=process_start):
    require(receipt.get('format') == 'CUDA-FOREGROUND-LEASE/1', 'wrong receipt format')
    require(owner is not None, 'native owner missing')
    require(receipt.get('state') == 'active' and owner.get('state') == 'active', 'lease is not active')
    require(owner.get('owner_kind') == 'foreground', 'not a foreground reservation')
    require(receipt.get('owner_id') == owner.get('id'), 'owner identity mismatch')
    require(not owner.get('preempt_requested'), 'owner preemption requested')
    expected = Path(expected_project).resolve()
    for record in (receipt, owner):
        require(bool(record.get('project_root')), 'project root absent')
        require(Path(record['project_root']).resolve() == expected, 'project root mismatch')
    pid = owner.get('pid')
    require(type(pid) is int and pid > 0 and receipt.get('pid') == pid, 'PID mismatch')
    require(bool(owner.get('process_start')), 'native process start absent')
    require(read_start(pid) == str(owner['process_start']), 'owner PID was reused or is stale')
    required = {'accelerator:' + uuid for uuid in expected_gpus}
    require(bool(required), 'expected GPU set empty')
    receipt_resources = set(receipt.get('resource_ids', []))
    owner_resources = set(owner.get('resources', []))
    require(required <= receipt_resources and required <= owner_resources, 'required GPU not reserved')
    require(receipt_resources <= owner_resources, 'receipt contains unreserved resources')
    return {'status': 'verified_active_native_gpu_lease', 'owner_id': owner['id'],
            'pid': pid, 'process_start': str(owner['process_start']),
            'project_root': str(expected), 'gpu_uuids': sorted(expected_gpus),
            'authority': 'HostCoordinator(create=False).owner'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease-receipt', required=True)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--gpu-uuid', action='append', required=True)
    parser.add_argument('--workflow-integration', default='/home/tumlinson/.agents/skills/integrations/coding-workflow-mcp')
    args = parser.parse_args()
    sys.path.insert(0, args.workflow_integration)
    from coding_workflow_mcp.runtime_identity import bind_canonical_runtime
    bind_canonical_runtime()
    from todo_orchestrator.background.host import HostCoordinator
    receipt = json.loads(Path(args.lease_receipt).read_text())
    owner_id = receipt.get('owner_id')
    require(isinstance(owner_id, str) and bool(owner_id), 'owner identity absent')
    owner = HostCoordinator(create=False).owner(owner_id)
    print(json.dumps(verify(receipt, owner, args.project_root, args.gpu_uuid), sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
        raise SystemExit('verify_gpu_lease: ' + str(error))
