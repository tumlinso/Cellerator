#!/usr/bin/env python3
"""Reject stale inputs and swapped executables before any device execution."""
import hashlib,json,pathlib,tempfile,unittest
import check_mma
import verify_gpu_evidence

class BuildBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temporary.name)
        self.manifest={'schema':'MMA-FRESH-BUILD/1','fresh_objects':True,
                       'source_hashes':check_mma.source_hashes(),
                       'input_hashes':check_mma.build_input_hashes(),'binaries':{}}
        for name,binary,_,_ in check_mma.binaries(self.root):
            binary.parent.mkdir(parents=True,exist_ok=True);binary.write_bytes(('fake-'+name).encode())
            self.manifest['binaries'][name]={'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
        self.save()
    def tearDown(self):self.temporary.cleanup()
    def save(self):check_mma.save_json(self.root/'build-manifest.json',self.manifest)
    def test_matching_identity(self):self.assertEqual(check_mma.load_build_manifest(self.root)[0],self.manifest)
    def test_false_fresh_objects(self):
        self.manifest['fresh_objects']=False;self.save()
        with self.assertRaisesRegex(AssertionError,'fresh_objects=true'):check_mma.load_build_manifest(self.root)
    def test_missing_fresh_objects(self):
        del self.manifest['fresh_objects'];self.save()
        with self.assertRaisesRegex(AssertionError,'fresh_objects=true'):check_mma.load_build_manifest(self.root)
    def test_stale_source(self):
        self.manifest['input_hashes']['CMakeLists.txt']='0'*64;self.save()
        with self.assertRaisesRegex(AssertionError,'inputs changed'):check_mma.load_build_manifest(self.root)
    def test_swapped_binary(self):
        check_mma.binaries(self.root)[0][1].write_bytes(b'older-binary')
        with self.assertRaisesRegex(AssertionError,'binary differs'):check_mma.load_build_manifest(self.root)
    def test_missing_manifest(self):
        (self.root/'build-manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):check_mma.load_build_manifest(self.root)

class CapabilityBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.results=pathlib.Path(self.temporary.name)
        self.manifest={'schema':'MMA-FRESH-BUILD/1','fresh_objects':True,
                       'input_hashes':check_mma.build_input_hashes(),
                       'source_hashes':check_mma.source_hashes(),
                       'binaries':{name:{'sha256':hashlib.sha256(('binary-'+name).encode()).hexdigest()}
                                   for name in ('patch16','quad','product')}}
        self.manifest_hash='a'*64
    def tearDown(self):self.temporary.cleanup()
    def module(self,name,manifest_hash=None):
        return {'module':name,'stage':'gpu','passed':True,'exit_code':0,
                'source_hashes':check_mma.source_hashes(name),
                'build_manifest_sha256':manifest_hash or self.manifest_hash,
                'binary_sha256':self.manifest['binaries'][name]['sha256']}
    def compiled_module(self,name):
        item=self.module(name);item['stage']='compiled_admission';return item
    def record(self,modules,manifest_hash=None):
        return {'stage':'gpu','passed':True,'source_hashes':check_mma.source_hashes(),
                'build_manifest_sha256':manifest_hash or self.manifest_hash,'modules':modules}
    def compiled_record(self,modules):
        item=self.record(modules);item['stage']='compiled_admission';return item
    def save(self,name,value):
        path=self.results/name;path.write_text(json.dumps(value));return path
    def test_current_build_and_receipts_preserve_gpu_status(self):
        modules={name:self.module(name) for name in ('patch16','quad','product')}
        for name,item in modules.items():self.save('gpu-'+name+'.json',item)
        self.save('gpu.json',self.record(modules))
        self.save('gpu-product-aggregate.json',self.record({'product':modules['product']}))
        validation={'gpu':{'passed':True,'evidence':'results/gpu.json'},
                    'gpu-product':{'passed':True,'evidence':'results/gpu-product-aggregate.json'}}
        current=check_mma.reconcile_gpu_validation(validation,self.results,self.manifest,self.manifest_hash)
        self.assertEqual(current,{name:True for name in ('patch16','quad','product')})
        self.assertIn('gpu',validation);self.assertIn('gpu-product',validation)
    def test_current_build_and_receipts_preserve_compiled_status(self):
        modules={name:self.compiled_module(name) for name in ('patch16','quad','product')}
        for name,item in modules.items():self.save('compiled_admission-'+name+'.json',item)
        self.save('compiled_admission.json',self.compiled_record(modules))
        self.save('compiled_admission-product-aggregate.json',self.compiled_record({'product':modules['product']}))
        validation={'compiled_admission':{'passed':True,'evidence':'results/compiled_admission.json'},
                    'compiled_admission-product':{'passed':True,'evidence':'results/compiled_admission-product-aggregate.json'}}
        check_mma.reconcile_gpu_validation(validation,self.results,self.manifest,self.manifest_hash)
        self.assertIn('compiled_admission',validation);self.assertIn('compiled_admission-product',validation)
    def test_stale_manifest_removes_passed_gpu_status(self):
        modules={name:self.module(name,'b'*64) for name in ('patch16','quad','product')}
        for name,item in modules.items():self.save('gpu-'+name+'.json',item)
        self.save('gpu.json',self.record(modules,'b'*64))
        self.save('gpu-product-aggregate.json',self.record({'product':modules['product']},'b'*64))
        validation={'gpu':{'passed':True,'evidence':'results/gpu.json'},
                    'gpu-product':{'passed':True,'evidence':'results/gpu-product-aggregate.json'},
                    'compiled_admission':{'passed':True,'evidence':'results/compiled_admission.json'},
                    'compiled_admission-product':{'passed':True,'evidence':'results/compiled_admission-product-aggregate.json'}}
        current=check_mma.reconcile_gpu_validation(validation,self.results,self.manifest,self.manifest_hash)
        self.assertEqual(current,{name:False for name in ('patch16','quad','product')})
        self.assertEqual(validation,{})
    def test_cpu_record_cannot_retain_gpu_status_without_current_manifest(self):
        validation={'gpu':{'passed':True,'evidence':'results/gpu.json'},
                    'compiled_admission':{'passed':True,'evidence':'results/compiled_admission.json'}}
        current=check_mma.reconcile_gpu_validation(validation,self.results,None,None)
        self.assertEqual(current,{name:False for name in ('patch16','quad','product')})
        self.assertEqual(validation,{})
    def test_saved_manifest_is_ignored_after_build_inputs_change(self):
        check_mma.save_json(self.results/'build-manifest.json',self.manifest)
        current,digest=check_mma.current_saved_manifest(self.results)
        self.assertEqual(current,self.manifest)
        self.assertEqual(digest,hashlib.sha256((self.results/'build-manifest.json').read_bytes()).hexdigest())
        self.manifest['input_hashes']['check_mma.py']='0'*64
        check_mma.save_json(self.results/'build-manifest.json',self.manifest)
        self.assertEqual(check_mma.current_saved_manifest(self.results),(None,None))
    def test_controller_receipt_must_bind_manifest_path_and_hash(self):
        manifest=self.results/'build-manifest.json';manifest.write_text('{}')
        digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
        receipt={'binaries':[{'path':str(manifest.resolve()),'sha256':digest}]}
        verify_gpu_evidence.require_manifest_receipt(receipt,manifest,digest)
        with self.assertRaisesRegex(ValueError,'does not bind'):
            verify_gpu_evidence.require_manifest_receipt({'binaries':[]},manifest,digest)
        with self.assertRaisesRegex(ValueError,'does not bind'):
            verify_gpu_evidence.require_manifest_receipt({'binaries':[{'path':str(manifest.resolve()),'sha256':'0'*64}]},manifest,digest)

if __name__=='__main__':unittest.main()
