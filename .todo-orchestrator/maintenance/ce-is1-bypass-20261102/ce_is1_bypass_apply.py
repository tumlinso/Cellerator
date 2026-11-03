"""One explicit, audited CE-IS1 operator override; no runtime or source edits."""
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

os.umask(0o077)
ROOT = Path('/home/tumlinson/Cellerator')
DIRECTORY = ROOT / '.todo-orchestrator/maintenance/ce-is1-bypass-20261102'
OPERATION = 'operator-ce-is1-bypass-20261102'
INSTRUCTION = 'Bypass all CE-IS1. You can edit sqlite directly if necessary.'

def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

from project_control.config import load_config
from project_control.native_workflow import local_mutator_workflow
from project_control.workflow_binding import initialize_workflow_binding

initialize_workflow_binding().validate()
from todo_orchestrator.contracts import EntityKind, EntityRef, StableId
from todo_orchestrator.authority import SQLiteAuthority, load_plan_revision_snapshot, store_plan_revision_snapshot
from todo_orchestrator.authority.commands import MutationContext
from todo_orchestrator.planning.service import _snapshot

spec = importlib.util.spec_from_file_location('ce_is1_bypass_plan', '/tmp/ce_is1_bypass_plan.py')
utility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility)

def targets(connection):
    return [dict(row) for row in connection.execute(
        "SELECT import_id,source_table,source_id,native_work_item_id,definition_revision_id,detail_json "
        "FROM legacy_entity_links WHERE source_table='tasks' AND source_id GLOB 'CE-IS1-*' ORDER BY import_id,source_id"
    )]

def preserved_rows(connection):
    tables = ('work_items','requirements','attempts','outcomes','outcome_candidates','outcome_provenance',
              'runs','run_lanes','assignments','run_queue','legacy_completion_assertions','legacy_history')
    result = {}
    for table in tables:
        rows = [dict(row) for row in connection.execute('SELECT * FROM "' + table + '" ORDER BY rowid')]
        result[table] = {'rows':len(rows), 'sha256':digest(rows)}
    return result

mode = sys.argv[1]
assert mode in ('preview','apply')
host = local_mutator_workflow(load_config())
_, session = host._session('cellerator')
try:
    snapshot = session.read_snapshot()
    model = session.read_model(expected_revision=snapshot.authority_revision).to_dict()
    project = session.project
    path = ROOT / '.todo-orchestrator/authority.sqlite3'
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as reader:
        reader.row_factory = sqlite3.Row
        old_links = targets(reader)
        actor_row = reader.execute(
            "SELECT actor_id,actor_incarnation_id FROM command_receipts WHERE command_kind='todo.run.retire' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert actor_row is not None
    actor = EntityRef(project.project_id, EntityKind.ACTOR, StableId(actor_row['actor_id']),
                      incarnation_id=StableId(actor_row['actor_incarnation_id']))
    record, metadata = utility.build_retired_snapshot(snapshot, actor, OPERATION)
    unfinished = [row for row in old_links if json.loads(row['detail_json'])['source_status'] != 'done']
    assert len(old_links)==20 and len(unfinished)==6, (len(old_links),len(unfinished))
    assert not [row for row in model['sections']['attempts'] if row['state'] not in ('accepted','rejected','released')]
    prepared = {'operation_id':OPERATION, 'instruction':INSTRUCTION, 'expected_revision':snapshot.authority_revision,
                'previous_plan_revision':str(snapshot.accepted_revision_id), 'new_plan_revision':str(record.plan_revision_id),
                'snapshot_digest':record.snapshot_digest, 'metadata':metadata, 'legacy_links':old_links,
                'unfinished_ids':[row['source_id'] for row in unfinished]}
    if mode=='preview':
        assert not DIRECTORY.exists(), 'Maintenance directory already exists; inspect rather than overwrite'
        DIRECTORY.mkdir(parents=True,mode=0o700)
        backup = session.backup(DIRECTORY/'authority-before.sqlite3',expected_revision=snapshot.authority_revision)
        (DIRECTORY/'backup-manifest.json').write_text(backup.manifest.to_json()+'\n')
        (DIRECTORY/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
        print(json.dumps({'status':'prepared','directory':str(DIRECTORY),'expected_revision':snapshot.authority_revision,
                          'new_plan_revision':str(record.plan_revision_id),'unfinished_ids':prepared['unfinished_ids'],
                          'metadata':metadata},indent=2))
        raise SystemExit(0)
    previous = json.loads((DIRECTORY/'prepared.json').read_text())
    assert prepared==previous, 'Authority changed after preview; no automatic retry'
finally:
    host._close(session)

with SQLiteAuthority.open_writer(path, project) as authority:
    with authority.write_transaction() as connection:
        assert connection.execute('SELECT semantic_revision FROM authority_meta WHERE singleton=1').fetchone()[0]==prepared['expected_revision']
        live_snapshot = _snapshot(connection, project)
        assert str(live_snapshot.accepted_revision_id)==prepared['previous_plan_revision']
        live_record, live_metadata = utility.build_retired_snapshot(live_snapshot, actor, OPERATION)
        assert live_record.snapshot_digest==prepared['snapshot_digest'] and live_metadata==metadata
        assert targets(connection)==old_links
        schema_before = [tuple(row) for row in connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name")]
        preserved_before = preserved_rows(connection)
        imports_before = [tuple(row) for row in connection.execute('SELECT import_id,source_digest,envelope_json FROM legacy_imports ORDER BY import_id')]
        non_target_links = [tuple(row) for row in connection.execute("SELECT * FROM legacy_entity_links WHERE NOT(source_table='tasks' AND source_id GLOB 'CE-IS1-*') ORDER BY import_id,source_table,source_id")]
        context = MutationContext(connection,project,actor,prepared['expected_revision'],authority,StableId(OPERATION))
        stored = store_plan_revision_snapshot(context,live_record,based_on_revision_id=live_snapshot.accepted_revision_id)
        context._deactivate()
        trigger_sql = connection.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='legacy_entity_links_no_update'").fetchone()[0]
        connection.execute('DROP TRIGGER legacy_entity_links_no_update')
        now = datetime.now(timezone.utc).isoformat(timespec='microseconds')
        changed_links=[]
        for row in unfinished:
            original=json.loads(row['detail_json']);updated=dict(original)
            updated['source_status']='cancelled'
            updated['operator_override']={'disposition':'bypassed','operation_id':OPERATION,'reason':INSTRUCTION,
                                         'occurred_at':now,'original_import_detail':original,
                                         'source_status_is_operator_override':True}
            replacement=canonical(updated)
            cursor=connection.execute(
                'UPDATE legacy_entity_links SET detail_json=? WHERE import_id=? AND source_table=? AND source_id=? AND detail_json=?',
                (replacement,row['import_id'],row['source_table'],row['source_id'],row['detail_json']))
            assert cursor.rowcount==1
            changed_links.append({'import_id':row['import_id'],'source_id':row['source_id'],
                                  'before_detail_json':row['detail_json'],'after_detail_json':replacement})
        connection.execute(trigger_sql)
        assert [tuple(row) for row in connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name")]==schema_before
        assert preserved_rows(connection)==preserved_before
        assert [tuple(row) for row in connection.execute('SELECT import_id,source_digest,envelope_json FROM legacy_imports ORDER BY import_id')]==imports_before
        assert [tuple(row) for row in connection.execute("SELECT * FROM legacy_entity_links WHERE NOT(source_table='tasks' AND source_id GLOB 'CE-IS1-*') ORDER BY import_id,source_table,source_id")]==non_target_links
        next_revision=prepared['expected_revision']+1
        result={'operation_id':OPERATION,'route':'explicit_user_authorized_sqlite_override',
                'previous_plan_revision':prepared['previous_plan_revision'],'plan_revision_id':str(stored.plan_revision_id),
                'retired_work_items':metadata,'cancelled_legacy_ids':prepared['unfinished_ids'],
                'records_preserved':True,'backup':str(DIRECTORY/'authority-before.sqlite3')}
        event_payload={**result,'human_instruction':INSTRUCTION,'legacy_status_overrides':changed_links,
                       'actor_attribution':'User-directed Codex maintenance using the existing Project Control mutator actor'}
        event_id='event-'+uuid4().hex;receipt_id='receipt-'+uuid4().hex
        assert connection.execute('UPDATE authority_meta SET semantic_revision=? WHERE singleton=1 AND semantic_revision=?',
                                  (next_revision,prepared['expected_revision'])).rowcount==1
        connection.execute('INSERT INTO authority_events(event_id,semantic_revision,event_ordinal,event_kind,payload_json,occurred_at,actor_id,actor_incarnation_id) VALUES (?,?,0,?,?,?,?,?)',
                           (event_id,next_revision,'operator.ce_is1.bypassed',canonical(event_payload),now,str(actor.entity_id),str(actor.incarnation_id)))
        connection.execute('INSERT INTO command_receipts(receipt_id,command_kind,semantic_revision,changed,event_id,actor_id,actor_incarnation_id,result_json,created_at) VALUES (?,?,?,1,?,?,?,?,?)',
                           (receipt_id,'operator.ce_is1.bypass',next_revision,event_id,str(actor.entity_id),str(actor.incarnation_id),canonical(result),now))
        connection.execute('INSERT INTO command_idempotency(idempotency_key,request_digest,receipt_id) VALUES (?,?,?)',
                           (OPERATION,digest(prepared),receipt_id))
        assert not connection.execute('PRAGMA foreign_key_check').fetchall()
        assert connection.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        loaded=load_plan_revision_snapshot(connection,project,stored.plan_revision_id)
        assert loaded.snapshot_digest==live_record.snapshot_digest
        assert connection.execute("SELECT COUNT(*) FROM plan_revision_work_items WHERE plan_revision_id=? AND work_item_id GLOB 'CE-IS1-*'",(str(stored.plan_revision_id),)).fetchone()[0]==0
        report={**result,'revision':next_revision,'event_id':event_id,'receipt_id':receipt_id,
                'schema_restored':True,'preserved_tables':preserved_before,'foreign_key_check':'ok','quick_check':'ok'}
(DIRECTORY/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
