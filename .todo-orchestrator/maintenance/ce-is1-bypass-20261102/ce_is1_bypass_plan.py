"""Prepare a source-bound CE-IS1 retirement snapshot; performs no writes.

Call build_retired_snapshot with a freshly read Todo AcceptedSnapshot, a
registered operator actor reference, and a unique operator operation ID.
The caller owns authorization, transaction, revision CAS, durable audit,
legacy-link handling, and persistence; this module never opens an authority.
"""
from dataclasses import replace
import hashlib
import json

from todo_orchestrator.contracts import EntityKind, EntityRef, StableId
from todo_orchestrator.planning.graph import validate_plan_graph
from todo_orchestrator.planning.snapshot import (
    accepted_plan_payload,
    decode_accepted_plan,
    definition_bundle_payloads,
    encode_accepted_snapshot,
)


def _is_target(identifier):
    value = str(identifier)
    return value == 'CE-IS1' or value.startswith('CE-IS1-')


def _ref_value(ref):
    return {'entity_id': str(ref.entity_id),
            'definition_revision_id': None if ref.definition_revision_id is None else str(ref.definition_revision_id),
            'kind': ref.kind.value, 'project_id': str(ref.project_id)}


def _bundle_map(plan):
    return {(item['entity_id'], item['definition_revision_id']):
            (item['canonical_json'], item['content_digest'])
            for item in definition_bundle_payloads(plan)}


def build_retired_snapshot(snapshot, actor, operation_id):
    """Build and strictly validate a new immutable accepted plan in memory."""
    plan = snapshot.accepted_plan
    if plan is None or snapshot.accepted_revision_id is None:
        raise ValueError('current accepted plan is required')
    if not isinstance(actor, EntityRef) or actor.kind is not EntityKind.ACTOR:
        raise ValueError('actor must be an exact registered actor reference')
    if actor.project_id != plan.project.project_id or actor.incarnation_id is None:
        raise ValueError('actor must bind an incarnation in the current project')
    operation_id = StableId(str(operation_id))
    work_removed = tuple(item for item in plan.work_items if _is_target(item.definition.work_item_id))
    work_ids = {str(item.definition.work_item_id) for item in work_removed}
    requirements_removed = tuple(item for item in plan.requirements
                                 if str(item.definition.work_item_id) in work_ids)
    requirement_ids = {str(item.definition.requirement_id) for item in requirements_removed}
    if len(work_ids) != 20 or len(requirements_removed) != 42 or len(requirement_ids) != 42:
        raise ValueError('current scope is not exactly 20 distinct CE-IS1 work items and 42 distinct owned requirements')
    removed_refs = {item.definition.ref for item in work_removed}
    for attempt in snapshot.attempts:
        if attempt.work_item in removed_refs and attempt.state.value in {
                'admitted', 'active', 'reported', 'expired', 'uncertain', 'quarantined'}:
            raise ValueError('retiring work still has an active or unresolved attempt')
    if any(outcome.work_item in removed_refs for outcome in snapshot.outcomes):
        raise ValueError('retiring work has a native accepted outcome; preserve its current membership')
    survivors = tuple(item for item in plan.work_items if str(item.definition.work_item_id) not in work_ids)
    requirements = tuple(item for item in plan.requirements
                         if str(item.definition.work_item_id) not in work_ids)

    def retired_source(group):
        ref = group.source.reference
        return (ref.project_id == plan.project.project_id and
                ((ref.kind is EntityKind.WORK_ITEM and str(ref.entity_id) in work_ids) or
                 (ref.kind is EntityKind.REQUIREMENT and str(ref.entity_id) in requirement_ids)))

    removed_groups = tuple(group for group in plan.blocking_groups if retired_source(group))
    groups = tuple(group for group in plan.blocking_groups if not retired_source(group))
    manifest = {
        'operation_id': str(operation_id), 'project_id': str(plan.project.project_id),
        'authority_id': str(plan.project.authority_id), 'authority_revision': snapshot.authority_revision,
        'parent_plan_revision_id': str(snapshot.accepted_revision_id),
        'retired_work_refs': sorted((_ref_value(x.definition.ref) for x in work_removed), key=lambda x:x['entity_id']),
        'retired_requirement_refs': sorted((_ref_value(x.definition.ref) for x in requirements_removed), key=lambda x:x['entity_id']),
        'actor': {'entity_id': str(actor.entity_id), 'incarnation_id': str(actor.incarnation_id)},
    }
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    manifest_digest = hashlib.sha256(manifest_json.encode('ascii')).hexdigest()
    new_id = StableId('accepted-' + manifest_digest)
    revision = replace(plan.revision, plan_revision_id=new_id, sequence=plan.revision.sequence + 1,
                       based_on_revision_id=snapshot.accepted_revision_id,
                       work_items=tuple(item.definition.ref for item in survivors))
    retired_plan = replace(plan, revision=revision, work_items=survivors,
                           requirements=requirements, blocking_groups=groups)
    graph = validate_plan_graph(retired_plan)
    if not graph.ok:
        raise ValueError('surviving graph validation failed: ' + '; '.join(x.code for x in graph.diagnostics))
    remaining_references = []
    historical_supersedes = []
    retired_revisions = {str(item.definition.work_item_id): str(item.definition.definition_revision_id)
                         for item in work_removed}
    survivor_revisions = {str(item.definition.work_item_id): str(item.definition.definition_revision_id)
                          for item in survivors}

    def is_retired_node(node):
        return (node.project_id == str(plan.project.project_id) and
                ((node.kind == 'work_item' and node.entity_id in work_ids) or
                 (node.kind == 'requirement' and node.entity_id in requirement_ids)))

    for edge in (*graph.edges, *graph.lineage_edges):
        source_retired = is_retired_node(edge.source)
        target_retired = is_retired_node(edge.target)
        if not source_retired and not target_retired:
            continue
        is_preserved_lineage = (
            edge in graph.lineage_edges and edge.relation == 'supersedes' and
            not source_retired and target_retired and
            edge.source.project_id == str(plan.project.project_id) and
            edge.source.kind == 'work_item' and
            survivor_revisions.get(edge.source.entity_id) == edge.source.revision_id and
            edge.target.kind == 'work_item' and
            retired_revisions.get(edge.target.entity_id) == edge.target.revision_id
        )
        row = {'relation': edge.relation, 'path': edge.path,
               'source': {'kind': edge.source.kind, 'entity_id': edge.source.entity_id,
                          'revision_id': edge.source.revision_id},
               'target': {'kind': edge.target.kind, 'entity_id': edge.target.entity_id,
                          'revision_id': edge.target.revision_id}}
        if is_preserved_lineage:
            historical_supersedes.append(row)
        else:
            remaining_references.append(row)
    if remaining_references:
        raise ValueError('surviving plan has live references to retired items: ' + json.dumps(remaining_references, sort_keys=True))
    if len(historical_supersedes) != 5:
        raise ValueError('current scope does not have exactly five preserved historical supersedes edges')
    original_bundles = _bundle_map(plan)
    new_bundles = _bundle_map(retired_plan)
    expected_bundles = {key: value for key, value in original_bundles.items() if key[0] not in work_ids}
    if new_bundles != expected_bundles:
        changed = sorted(set(new_bundles) ^ set(expected_bundles) |
                         {key for key in set(new_bundles) & set(expected_bundles)
                          if new_bundles[key] != expected_bundles[key]})
        raise ValueError('surviving immutable work bundles changed: ' + repr(changed))
    record = encode_accepted_snapshot(retired_plan, authority_id=plan.project.authority_id,
                                      semantic_revision=snapshot.authority_revision + 1,
                                      based_on_revision_id=snapshot.accepted_revision_id)
    decoded = decode_accepted_plan(record.canonical_json, plan.project, sequence=revision.sequence)
    if accepted_plan_payload(decoded) != accepted_plan_payload(retired_plan):
        raise ValueError('accepted snapshot typed payload does not round-trip')
    if _bundle_map(decoded) != expected_bundles:
        raise ValueError('decoded surviving work bundles differ from original')
    roundtrip = encode_accepted_snapshot(decoded, authority_id=plan.project.authority_id,
                                         semantic_revision=snapshot.authority_revision + 1,
                                         based_on_revision_id=snapshot.accepted_revision_id)
    if (roundtrip.canonical_json, roundtrip.snapshot_digest) != (record.canonical_json, record.snapshot_digest):
        raise ValueError('accepted snapshot canonical bytes do not round-trip')
    metadata = {
        **manifest, 'operation_manifest_sha256': manifest_digest,
        'new_plan_revision_id': str(new_id), 'new_snapshot_sha256': record.snapshot_digest,
        'removed_work_count': len(work_ids), 'removed_requirement_count': len(requirement_ids),
        'removed_group_count': len(removed_groups), 'surviving_work_count': len(survivors),
        'surviving_requirement_count': len(requirements), 'surviving_group_count': len(groups),
        'surviving_bundle_count': len(expected_bundles), 'graph_node_count': len(graph.nodes),
        'graph_edge_count': len(graph.edges), 'graph_lineage_edge_count': len(graph.lineage_edges),
        'remaining_retired_references': remaining_references,
        'preserved_historical_supersedes': historical_supersedes,
        'validation': 'graph and explicit retired-reference scan pass; typed codec and surviving bundles match',
    }
    return record, metadata


def main():
    # This diagnostic obtains a read-only native snapshot through the trusted
    # installed Project Control host. It does not open a writer or commit.
    from project_control.config import load_config
    from project_control.native_workflow import local_mutator_workflow
    host = local_mutator_workflow(load_config())
    root, session = host._session('cellerator')
    try:
        with session._reader() as authority:
            snapshot = session._services.plan.read_snapshot(authority)
        if not snapshot.attempts:
            raise ValueError('no registered prior actor is available for read-only validation')
        actor = snapshot.attempts[0].actor
        record, metadata = build_retired_snapshot(snapshot, actor, 'ce-is1-read-only-validation')
        print(json.dumps({'status':'ok','metadata':metadata}, sort_keys=True, indent=2))
    finally:
        host._close(session)


if __name__ == '__main__':
    main()
