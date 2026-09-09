#!/usr/bin/env python3
"""Adversarial policy tests, never evidence of a held physical GPU lease."""
import copy
import os
import unittest
from verify_gpu_lease import verify, process_start


class LeasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.receipt = dict(format='CUDA-FOREGROUND-LEASE/1', owner_id='test-owner',
                            project_root='/tmp/project', pid=123, state='active',
                            resource_ids=['accelerator:GPU-test'])
        self.owner = dict(id='test-owner', owner_kind='foreground', project_root='/tmp/project',
                          pid=123, process_start='456', state='active', preempt_requested=0,
                          resources=['accelerator:GPU-test'])

    def check(self, receipt=None, owner=None, start='456'):
        return verify(receipt or self.receipt, owner or self.owner, '/tmp/project',
                      ['GPU-test'], lambda pid: start)

    def test_matching_policy(self):
        self.assertEqual(self.check()['status'], 'verified_active_native_gpu_lease')

    def test_live_process_start(self):
        self.assertTrue(process_start(os.getpid()).isdigit())

    def test_mismatched_fields(self):
        for key, value in [('state', 'released'), ('id', 'other'), ('pid', 124),
                           ('project_root', '/tmp/other'), ('resources', []),
                           ('owner_kind', 'background'), ('preempt_requested', 1),
                           ('process_start', None)]:
            with self.subTest(key=key):
                owner = copy.deepcopy(self.owner); owner[key] = value
                with self.assertRaises(ValueError): self.check(owner=owner)

    def test_stale_pid(self):
        with self.assertRaises(ValueError): self.check(start='457')

    def test_missing_owner(self):
        with self.assertRaises(ValueError):
            verify(self.receipt, None, '/tmp/project', ['GPU-test'])

    def test_forged_receipt(self):
        for key, value in [('format', 'other'), ('owner_id', 'other'), ('pid', 124),
                           ('project_root', '/tmp/other'), ('resource_ids', ['accelerator:GPU-foreign'])]:
            with self.subTest(key=key):
                receipt = copy.deepcopy(self.receipt); receipt[key] = value
                with self.assertRaises(ValueError): self.check(receipt=receipt)

    def test_dead_pid(self):
        def dead(pid): raise FileNotFoundError('dead PID')
        with self.assertRaises(FileNotFoundError):
            verify(self.receipt, self.owner, '/tmp/project', ['GPU-test'], dead)


if __name__ == '__main__': unittest.main()
