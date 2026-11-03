import argparse
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('cgs_register_grouping_probe', scripts / 'register_grouping_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
campaign = module.campaign


class RegistrationTest(unittest.TestCase):
    def test_mocked_registration_uses_nested_output_and_exact_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = root / 'study'
            (study / 'scripts').mkdir(parents=True)
            adapter = study / 'scripts/probe_packet.py'
            adapter.write_text('reviewed adapter')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = {'files': {'Cellerator/bench/chromatin_geometry/scripts/probe_packet.py': {'sha256': digest(adapter)}}}
            archive = root / 'source.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                raw = json.dumps(manifest).encode()
                info = tarfile.TarInfo('SOURCE-MANIFEST.json')
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
            study_binary = root / 'study.bin'
            qual_binary = root / 'qual.bin'
            study_binary.write_bytes(b'study')
            qual_binary.write_bytes(b'qualifier')
            receipts = []
            for index, binary in enumerate((study_binary, qual_binary)):
                receipt = root / f'receipt-{index}.json'
                receipt.write_text(json.dumps({'schema':'cgs.build.v1','archive_sha256':digest(archive),'binary_sha256':digest(binary)}))
                receipts.append(receipt)
            proof = root / 'proof.json'
            proof.write_text('{}')
            old_study, old_lanes, old_ledger = campaign.STUDY, campaign.LANES, campaign.LEDGER
            campaign.STUDY, campaign.LANES, campaign.LEDGER = study, study/'runs/lanes', study/'runs/campaign-budget.json'
            try:
                campaign.LEDGER.parent.mkdir(parents=True)
                for lane, phase in enumerate(('qualification', 'r1')):
                    lane_dir = campaign.LANES / str(lane)
                    (lane_dir / '.autobench').mkdir(parents=True)
                    (lane_dir / '.autobench/config.json').write_text('{"active":false}')
                    args = argparse.Namespace(lane_root=campaign.LANES,lane=lane,phase=phase,uuid='GPU-test',
                        resource_id='gpu:host:GPU-test',archive=archive,archive_sha256=digest(archive),
                        binary=study_binary,receipt=receipts[0],qualification_binary=qual_binary,
                        qualification_receipt=receipts[1],qualification_proof=proof if phase=='r1' else None)
                    with patch.object(campaign, 'command', return_value={'paused':False,'attempts':[]}), \
                         patch.object(module.subprocess, 'run', return_value=argparse.Namespace(returncode=0,stdout='registered',stderr='')):
                        module.register(args)
                    packet = tomllib.loads((lane_dir/'registration.toml').read_text())['packet'][0]
                    argv = packet['argv']
                    self.assertEqual(argv[argv.index('--output')+1], '{output}/probe-packet')
                    self.assertEqual(packet['kill_seconds'], 240 if phase=='qualification' else 90)
                    self.assertEqual(('--suite' in argv), phase!='qualification')
                    self.assertEqual(('--qualification-proof' in argv), phase!='qualification')
            finally:
                campaign.STUDY, campaign.LANES, campaign.LEDGER = old_study, old_lanes, old_ledger


if __name__ == '__main__':
    unittest.main()
