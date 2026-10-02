#!/usr/bin/env python3
"""Opt-in complete-cost CUDA comparison for the shared-support composition.

Run with the matching CelleraTorch library and an assigned GPU lease. Nothing
runs on CUDA without --run-cuda. Output is evidence, not a promotion decision.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-cuda', action='store_true', help='execute under an assigned CUDA lease')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=5, choices=range(5, 11))
    p.add_argument('--seed', type=int, default=1701)
    return p


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parser().parse_args()
    if not args.run_cuda:
        raise SystemExit('CUDA execution requires explicit --run-cuda and an assigned lease')
    import torch
    from celleratorch import (Axis, Identity, SharedSupportSpec,
                             SharedSupportRelation, guarded_step)
    if not torch.cuda.is_available():
        raise RuntimeError('assigned CUDA device is unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0')
    torch.cuda.synchronize(device)

    def timed(fn):
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        value = fn()
        torch.cuda.synchronize(device)
        return value, (time.perf_counter_ns() - start) / 1e6

    def axis(extent, offset):
        return Axis(*(Identity(offset + i, 1) for i in range(4)), extent)

    class Materialized(torch.nn.Module):
        def __init__(self, coefficients, src, dst, outputs):
            super().__init__()
            self.coefficients = torch.nn.Parameter(coefficients.clone())
            self.register_buffer('src', src)
            self.register_buffer('dst', dst)
            self.outputs = outputs

        def forward(self, x, s, a):
            # Explicit per-instance edge weights; repeated logical edges remain
            # distinct entries and index_add adds their output contributions.
            edges = (self.coefficients[None, :] * s[:, self.src]
                     * a[:, self.dst])
            values = edges * x[:, self.src]
            return torch.zeros(x.shape[0], self.outputs, device=x.device,
                               dtype=x.dtype).index_add(1, self.dst, values)

    def optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False,
                                fused=False, weight_decay=0)

    def equality(native, baseline, data):
        nx, ns, na = [v.to(device).requires_grad_() for v in data[:3]]
        bx, bs, ba = [v.to(device).requires_grad_() for v in data[:3]]
        target = data[3].to(device)
        ny, by = native(nx, ns, na), baseline(bx, bs, ba)
        torch.testing.assert_close(ny, by, rtol=2e-5, atol=2e-6)
        ((ny - target).square().mean()).backward()
        ((by - target).square().mean()).backward()
        gradients = [(nx.grad, bx.grad), (ns.grad, bs.grad), (na.grad, ba.grad),
                     (native.coefficients.grad, baseline.coefficients.grad)]
        for lhs, rhs in gradients:
            torch.testing.assert_close(lhs, rhs, rtol=3e-5, atol=3e-6)
        no, bo = optimizer(native), optimizer(baseline)
        guarded_step(native, no)
        bo.step()
        torch.testing.assert_close(native.coefficients, baseline.coefficients,
                                   rtol=3e-5, atol=3e-6)
        return {'forward_max_abs': float((ny - by).abs().max()),
                'gradient_max_abs': [float((l-r).abs().max()) for l,r in gradients],
                'adam_parameter_max_abs': float((native.coefficients-baseline.coefficients).abs().max())}

    results = []
    for name, batch, sources, outputs, edges in [
        ('shared_batch', 32, 64, 32, 256), ('small_counter_regime', 1, 64, 32, 16)
    ]:
        gen = torch.Generator().manual_seed(args.seed + batch)
        src_cpu = torch.randint(sources, (edges,), generator=gen)
        dst_cpu = torch.randint(outputs, (edges,), generator=gen)
        weights_cpu = torch.randn(edges, generator=gen) * .05
        data = (torch.randn(batch, sources, generator=gen),
                torch.randn(batch, sources, generator=gen),
                torch.randn(batch, outputs, generator=gen),
                torch.randn(batch, outputs, generator=gen))

        def make_spec():
            return SharedSupportSpec(axis(sources, 10), axis(outputs, 20),
                axis(edges, 30), tuple(Identity(1000+i, 2) for i in range(edges)),
                tuple(src_cpu.tolist()), tuple(dst_cpu.tolist()))

        spec = make_spec()
        weights = weights_cpu.to(device)
        native = SharedSupportRelation(spec, weights, max_batch=batch,
                                       max_live_forwards=2)
        baseline = Materialized(weights, src_cpu.to(device), dst_cpu.to(device), outputs)
        reference = equality(native, baseline, data)
        del native, baseline, weights
        gc.collect()
        torch.cuda.empty_cache()
        paths = {}
        for kind in ('native_composition', 'torch_materialized'):
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = torch.cuda.memory_allocated(device)
            free_before, device_total = torch.cuda.mem_get_info(device)
            observed_free = [free_before]
            wall_start = time.perf_counter_ns()
            packed, packing_ms = timed(make_spec)
            initial, weight_upload_ms = timed(lambda: weights_cpu.to(device))
            if kind == 'native_composition':
                # Constructor includes semantic lowering, native support upload,
                # program preparation and native-owned coefficient allocation.
                model, preparation_ms = timed(lambda: SharedSupportRelation(
                    packed, initial, max_batch=batch, max_live_forwards=2))
                support_upload_ms = None  # included in preparation, cannot separate publicly
            else:
                support, support_upload_ms = timed(lambda: (src_cpu.to(device), dst_cpu.to(device)))
                model, preparation_ms = timed(lambda: Materialized(initial, *support, outputs))
            opt, optimizer_setup_ms = timed(lambda: optimizer(model))
            torch.cuda.synchronize(device)
            cold_total_ms = (time.perf_counter_ns() - wall_start) / 1e6
            free_after_setup, _ = torch.cuda.mem_get_info(device)
            observed_free.append(free_after_setup)
            samples = []
            for iteration in range(args.repetitions + 2):
                start = time.perf_counter_ns()
                opt.zero_grad(set_to_none=True)
                tensors, upload_ms = timed(lambda: [v.to(device) for v in data])
                x, s, a, target = tensors
                x.requires_grad_(); s.requires_grad_(); a.requires_grad_()
                y, forward_ms = timed(lambda: model(x, s, a))
                loss, loss_ms = timed(lambda: (y-target).square().mean())
                _, backward_ms = timed(loss.backward)
                _, step_ms = timed(lambda: guarded_step(model, opt)
                                   if kind == 'native_composition' else opt.step())
                torch.cuda.synchronize(device)
                sample = {'input_target_upload_ms': upload_ms, 'forward_ms': forward_ms,
                          'loss_ms': loss_ms, 'backward_ms': backward_ms,
                          'adam_and_publication_ms': step_ms,
                          'complete_iteration_ms': (time.perf_counter_ns()-start)/1e6}
                observed_free.append(torch.cuda.mem_get_info(device)[0])
                if iteration >= 2:
                    samples.append(sample)
                del y, loss, tensors, x, s, a, target
            paths[kind] = {
                'cold': {'support_identity_packing_ms': packing_ms,
                         'initial_coefficients_upload_ms': weight_upload_ms,
                         'support_upload_ms': support_upload_ms,
                         'program_or_module_preparation_ms': preparation_ms,
                         'optimizer_setup_ms': optimizer_setup_ms,
                         'complete_setup_ms': cold_total_ms},
                'median_ms': {key: statistics.median(s[key] for s in samples) for key in samples[0]},
                'samples': samples,
                'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'torch_allocated_before_bytes': memory_before,
                'native_reserved_bytes': None,
                'device_memory_observation': {
                    'total_bytes': device_total, 'free_before_setup_bytes': free_before,
                    'free_after_setup_bytes': free_after_setup,
                    'setup_global_used_delta_bytes': free_before-free_after_setup,
                    'max_observed_global_used_delta_bytes': free_before-min(observed_free),
                    'free_bytes_at_setup_and_iteration_boundaries': observed_free,
                    'note': 'Global device observations include native allocations and Torch pools, may include other processes, and miss within-stage transients. They are not per-owner reserved bytes and must not be summed with Torch counters.'},
                'native_memory_note': 'No public native allocation-byte query; Torch peaks exclude raw native CUDA allocations.'}
            del model, opt, initial, packed
            if kind == 'torch_materialized':
                del support
            gc.collect()
            torch.cuda.empty_cache()
        results.append({'name': name, 'batch': batch, 'source_extent': sources,
                        'target_extent': outputs, 'logical_edges': edges,
                        'reference_equality': reference, 'paths': paths,
                        'theoretical_value_storage_bytes': {
                            'shared_edge_values': 4*edges,
                            'activities': 4*batch*(sources+outputs),
                            'materialized_instance_edges': 4*batch*edges,
                            'shared_int64_topology': 16*edges},
                        'storage_note': 'Array formulas only; exclude optimizer, saved activations, identities and prepared native workspaces.'})
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__), root/'components/CelleraTorch/python/celleratorch/biology.py',
                    root/'components/CelleraTorch/python/celleratorch/mechanism.py',
                    root/'components/CelleraTorch/src/mechanism.cc']
    library = Path(os.environ['CELLERATORCH_NATIVE_LIBRARY']).resolve()
    output = {'schema_version': 1, 'disposition': 'evaluated_not_promoted',
              'precision': 'f32', 'seed': args.seed, 'repetitions': args.repetitions,
              'warmup_iterations': 2, 'results': results,
              'timing_policy': 'Synchronized wall times per stage and complete iteration; not unsynchronized kernel timings. Cold means fresh program, after library/context warmup.',
              'checkpoint_rebuild': 'not measured',
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                  'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()},
              'source_sha256': {str(p.relative_to(root)): digest(p) for p in source_paths},
              'native_library': {'path': str(library), 'sha256': digest(library)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'disposition': output['disposition']}))


if __name__ == '__main__':
    main()
