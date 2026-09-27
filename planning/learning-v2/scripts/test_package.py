#!/usr/bin/env python3
"""Regressions for the package linter only; not canonical workflow tests."""
import copy
import json
from pathlib import Path
import unittest
from check_package import lint_plan, PackageError
ROOT = Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def setUp(self): self.plan = json.loads((ROOT/'machine/glasshelix.todo-plan.json').read_text())
    def reject(self, f):
        f(self.plan)
        with self.assertRaises(PackageError): lint_plan(self.plan, 'GH-ML2', 'GlassHelix')
    def test_valid_gh(self): self.assertEqual(lint_plan(self.plan, 'GH-ML2', 'GlassHelix')['outcomes'], 6)
    def test_valid_ce(self):
        p = json.loads((ROOT/'machine/cellerator.todo-plan.json').read_text())
        self.assertEqual(lint_plan(p, 'CE-ML2', 'Cellerator')['outcomes'], 2)
    def test_wrong_target(self): self.reject(lambda p: p['project'].update(name='Cellerator'))
    def test_duplicate(self): self.reject(lambda p: p['tasks'].append(copy.deepcopy(p['tasks'][1])))
    def test_foreign_dependency(self): self.reject(lambda p: p['tasks'][1]['depends_on'].append({'type':'task','task_id':'CE-ML2-TRAIN'}))
    def test_aggregate_dependency(self): self.reject(lambda p: p['tasks'][1]['depends_on'].append({'type':'task','task_id':'GH-ML2-0000'}))
    def test_path_escape(self): self.reject(lambda p: p['tasks'][1]['scope']['exclusive_paths'].append('../Cellerator'))
    def test_false_completion(self): self.reject(lambda p: p['tasks'][1].update(status='done'))
    def test_role_cannot_edit(self): self.reject(lambda p: p['runs'][0]['lanes'][0].update(role='coordinator'))
    def test_cycle(self): self.reject(lambda p: p['tasks'][1]['depends_on'].append({'type':'task','task_id':'GH-ML2-LEARN'}))
    def test_bad_queue(self): self.reject(lambda p: p['runs'][0]['lanes'][0]['tasks'].reverse())
    def test_fake_gate(self): self.reject(lambda p: p['tasks'][1].update(gates=[{'id':'fake','argv':['true']}]))
if __name__ == '__main__': unittest.main()
