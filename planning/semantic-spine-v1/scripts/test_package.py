#!/usr/bin/env python3
"""Offline self-tests only. All native-tool calls in tests are mocked, never executed."""
from __future__ import annotations
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
from compile_plan import assemble
from validate_package import validate
import todo_bootstrap as boot

P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]

class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/'package'
        shutil.copytree(P,self.root,ignore=shutil.ignore_patterns('__pycache__'))
    def tearDown(self):self.temp.cleanup()
    def master(self):return json.loads((self.root/'machine/proposed_todos.json').read_text())
    def set_master(self,m):
        (self.root/'machine/proposed_todos.json').write_text(json.dumps(m))
        (self.root/'machine/semantic-spine-v1.todo-plan.json').write_text(json.dumps(assemble(m)))
    def test_good_graph(self):
        r=validate(self.root,ROOT,False)
        self.assertEqual(r['leaf_tasks'],29)
        self.assertTrue(r['four_way_post_foundation_fanout'])
    def test_plan_catalog_divergence(self):
        p=self.root/'machine/semantic-spine-v1.todo-plan.json';m=json.loads(p.read_text());m['tasks'][1]['title']='tampered';p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'diverges'):validate(self.root,None,False)
    def test_cycle_detected(self):
        m=self.master();t=m['tasks'][0]
        t['depends_on']=['CE-SS1-C04'];t['native_record']['depends_on']=[{'type':'task','task_id':'CE-SS1-C04'}]
        self.set_master(m)
        with self.assertRaisesRegex(ValueError,'cycle'):validate(self.root,None,False)
    def test_downgrade_rejected(self):
        m=self.master();m['native_schema_version']=2
        with self.assertRaisesRegex(ValueError,'schema'):assemble(m)
    def test_duplicate_id(self):
        m=self.master();m['tasks'].append(copy.deepcopy(m['tasks'][0]));self.set_master(m)
        with self.assertRaisesRegex(ValueError,'duplicate Todo'):validate(self.root,None,False)
    def test_task_budget(self):
        m=self.master()
        for i in range(70):
            t=copy.deepcopy(m['tasks'][0]);t['id']=t['native_record']['id']=f'CE-SS1-EXTRA-{i}'
            m['tasks'].append(t)
        self.set_master(m)
        with self.assertRaisesRegex(ValueError,'budget'):validate(self.root,None,False)
    def test_multiple_lane_assignment(self):
        m=self.master();m['run']['lanes'][2]['tasks'].append('CE-SS1-F01');self.set_master(m)
        with self.assertRaisesRegex(ValueError,'multiple lanes'):validate(self.root,None,False)
    def test_cross_lane_scope_collision(self):
        m=self.master();a=next(t for t in m['tasks'] if t['short_id']=='N01');b=next(t for t in m['tasks'] if t['short_id']=='F01')
        a['write_scope'].append(b['write_scope'][0]);a['native_record']['scope']['exclusive_paths']=a['write_scope'];self.set_master(m)
        with self.assertRaisesRegex(ValueError,'scope collision'):validate(self.root,None,False)
    def test_missing_demo(self):
        with self.assertRaisesRegex(ValueError,'demo missing'):validate(self.root,Path(self.temp.name),False)
    def test_apply_schema_never_auto_starts(self):
        m=assemble(self.master())
        self.assertTrue(all(t['status']=='planned' for t in m['tasks']))
        self.assertNotIn('activate',m['runs'][0])
        self.assertEqual(len(m['runs']),1)

class NativeGuardTests(unittest.TestCase):
    def response(self):
        return {'valid':True,'project_uuid':boot.UUID,'revision':12,'plan_digest':'abc',
                'would_add':['task:X'],'would_modify':[],
                'current_observation_preconditions':{'workflow_authority_fingerprint':'workflow'}}
    def test_unavailable_authority_rejected(self):
        r=self.response();r['revision']=None
        with patch.object(boot,'invoke',return_value=r):
            with self.assertRaisesRegex(RuntimeError,'UUID/revision'):boot.native_preview(ROOT,'mock-executable')
    def test_schema_rejection_stops(self):
        r=self.response();r['valid']=False
        with patch.object(boot,'invoke',return_value=r):
            with self.assertRaisesRegex(RuntimeError,'valid=true'):boot.native_preview(ROOT,'mock-executable')
    def test_existing_record_modification_stops(self):
        r=self.response();r['would_modify']=['CE-SS1-C01']
        with patch.object(boot,'invoke',return_value=r):
            with self.assertRaisesRegex(RuntimeError,'modify existing'):boot.native_preview(ROOT,'mock-executable')
    def test_workflow_unavailable_stops(self):
        r=self.response();r['current_observation_preconditions']={}
        with patch.object(boot,'invoke',return_value=r):
            with self.assertRaisesRegex(RuntimeError,'workflow authority'):boot.native_preview(ROOT,'mock-executable')
    def test_incomplete_diff_rejected(self):
        r=self.response();del r['would_modify']
        with patch.object(boot,'invoke',return_value=r):
            with self.assertRaisesRegex(RuntimeError,'diff output'):boot.native_preview(ROOT,'mock-executable')
    def test_preview_uses_only_known_nonmutating_argv(self):
        r=self.response()
        with patch.object(boot,'invoke',return_value=r) as call:
            boot.native_preview(ROOT,'mock-executable')
            args=call.call_args.args[0]
            self.assertEqual(args[:3],['mock-executable','plan','validate'])
            self.assertNotIn('--json',args);self.assertNotIn('apply',args)
    def test_apply_wrong_confirmation_calls_no_native_mutator(self):
        with patch.object(sys,'argv',['todo_bootstrap.py','apply','--source-root',str(ROOT),'--receipt','/tmp/probe.json','--confirm','WRONG']), \
             patch.object(boot,'check_repo',return_value={}),patch.object(boot,'tool_path',return_value='mock'), \
             patch.object(boot,'invoke') as call:
            with self.assertRaisesRegex(RuntimeError,'--confirm'):boot.main()
            call.assert_not_called()
    def test_receipts_cannot_overwrite_package(self):
        with self.assertRaisesRegex(RuntimeError,'outside the package'):boot.save(P/'new-receipt.json',{})

class FixtureTests(unittest.TestCase):
    def test_compile_and_run_reference_fixture(self):
        compiler=shutil.which('c++') or shutil.which('g++')
        if compiler is None:self.fail('C++ compiler unavailable: reference-fixture self-test not executed')
        with tempfile.TemporaryDirectory() as d:
            exe=Path(d)/'fixture'
            cmd=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pedantic','-DCELLERATOR_SPINE_REFERENCE_ONLY=1',str(ROOT/'examples/semantic_spine_v1/regulatory_reuse.cc'),'-o',str(exe)]
            subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            r=subprocess.run([str(exe)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.assertIn('REFERENCE_ONLY_FIXTURE_PASS',r.stdout)
            self.assertNotIn('SPINE_DEMO_GPU_PASS',r.stdout)
            self.assertIn('Generation-2 forward: 1 0.75 -0.5 0 1.5',r.stdout)
            self.assertIn('Generation-2 transpose: 1.25 -3 0.75 -0.125',r.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
