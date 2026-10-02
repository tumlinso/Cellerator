#!/usr/bin/env python3
"""Reject stale inputs and swapped executables before any device execution."""
import hashlib,json,pathlib,tempfile,unittest
import check_mma

class BuildBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temporary.name)
        self.manifest={'schema':'MMA-FRESH-BUILD/1','source_hashes':check_mma.source_hashes(),
                       'input_hashes':check_mma.build_input_hashes(),'binaries':{}}
        for name,binary,_,_ in check_mma.binaries(self.root):
            binary.parent.mkdir(parents=True,exist_ok=True);binary.write_bytes(('fake-'+name).encode())
            self.manifest['binaries'][name]={'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
        self.save()
    def tearDown(self):self.temporary.cleanup()
    def save(self):check_mma.save_json(self.root/'build-manifest.json',self.manifest)
    def test_matching_identity(self):self.assertEqual(check_mma.load_build_manifest(self.root)[0],self.manifest)
    def test_stale_source(self):
        self.manifest['input_hashes']['CMakeLists.txt']='0'*64;self.save()
        with self.assertRaisesRegex(AssertionError,'inputs changed'):check_mma.load_build_manifest(self.root)
    def test_swapped_binary(self):
        check_mma.binaries(self.root)[0][1].write_bytes(b'older-binary')
        with self.assertRaisesRegex(AssertionError,'binary differs'):check_mma.load_build_manifest(self.root)
    def test_missing_manifest(self):
        (self.root/'build-manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):check_mma.load_build_manifest(self.root)

if __name__=='__main__':unittest.main()
