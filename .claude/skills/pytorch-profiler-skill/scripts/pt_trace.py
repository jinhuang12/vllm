#!/usr/bin/env python3
"""
PyTorch Profiler Trace Analysis Tool

Usage:
    python pt_trace.py trace.json.gz --summary
    python pt_trace.py trace.json.gz --top-kernels 20
    python pt_trace.py trace.json.gz --breakdown
    python pt_trace.py trace.json.gz --inductor
"""

import gzip
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Iterator, Literal


class PTTrace:
    """PyTorch Profiler trace analyzer."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self._data = None
        self._events = None
        self._kernels = None
        self._cpu_ops = None
        self._python_funcs = None
        self._cuda_runtime = None
        
    def _load(self):
        """Lazy load trace data."""
        if self._data is not None:
            return
        
        if self.path.suffix == '.gz':
            with gzip.open(self.path, 'rt') as f:
                self._data = json.load(f)
        else:
            with open(self.path, 'r') as f:
                self._data = json.load(f)
        
        self._events = self._data.get('traceEvents', [])
        
    def _index_events(self):
        """Build event indices by category."""
        self._load()
        if self._kernels is not None:
            return
            
        self._kernels = []
        self._cpu_ops = []
        self._python_funcs = []
        self._cuda_runtime = []
        self._user_annotations = []
        
        for ev in self._events:
            cat = ev.get('cat', '')
            if cat == 'kernel':
                self._kernels.append(ev)
            elif cat == 'cpu_op':
                self._cpu_ops.append(ev)
            elif cat == 'python_function':
                self._python_funcs.append(ev)
            elif cat == 'cuda_runtime':
                self._cuda_runtime.append(ev)
            elif cat == 'user_annotation':
                self._user_annotations.append(ev)
                
    # -------------------------------------------------------------------------
    # Device & Metadata
    # -------------------------------------------------------------------------
    
    def devices(self) -> list[dict]:
        """Get GPU device properties."""
        self._load()
        return self._data.get('deviceProperties', [])
    
    def distributed_info(self) -> dict:
        """Get distributed training info (NCCL, rank, world size)."""
        self._load()
        return self._data.get('distributedInfo', {})
    
    # -------------------------------------------------------------------------
    # Event Access
    # -------------------------------------------------------------------------
    
    def events(self, cat: Optional[str] = None) -> list[dict]:
        """Get events, optionally filtered by category."""
        self._index_events()
        if cat is None:
            return self._events
        elif cat == 'kernel':
            return self._kernels
        elif cat == 'cpu_op':
            return self._cpu_ops
        elif cat == 'python_function':
            return self._python_funcs
        elif cat == 'cuda_runtime':
            return self._cuda_runtime
        elif cat == 'user_annotation':
            return self._user_annotations
        else:
            return [e for e in self._events if e.get('cat') == cat]
    
    def kernels(self) -> list[dict]:
        """Get all GPU kernel events."""
        return self.events('kernel')
    
    def cpu_ops(self) -> list[dict]:
        """Get all CPU operator events."""
        return self.events('cpu_op')
    
    # -------------------------------------------------------------------------
    # Summary & Analysis
    # -------------------------------------------------------------------------
    
    def summary(self) -> dict:
        """Get executive summary of the trace."""
        self._index_events()
        
        devices = self.devices()
        dist = self.distributed_info()
        
        total_kernel_time = sum(k.get('dur', 0) for k in self._kernels)
        unique_kernels = len(set(k['name'] for k in self._kernels))
        
        return {
            'device': devices[0]['name'] if devices else 'Unknown',
            'num_gpus': len(devices),
            'distributed': {
                'backend': dist.get('backend'),
                'rank': dist.get('rank'),
                'world_size': dist.get('world_size'),
            } if dist else None,
            'total_gpu_time_ms': total_kernel_time / 1000,
            'total_kernel_launches': len(self._kernels),
            'unique_kernels': unique_kernels,
            'total_cpu_ops': len(self._cpu_ops),
        }
    
    def top_kernels(self, n: int = 20, sort_by: Literal['total', 'avg', 'count'] = 'total') -> list[dict]:
        """Get top N kernels by time or count."""
        self._index_events()
        
        stats = defaultdict(lambda: {'count': 0, 'total_us': 0})
        for k in self._kernels:
            name = k['name']
            stats[name]['count'] += 1
            stats[name]['total_us'] += k.get('dur', 0)
        
        total_time = sum(v['total_us'] for v in stats.values())
        
        results = []
        for name, s in stats.items():
            results.append({
                'name': name,
                'count': s['count'],
                'total_ms': s['total_us'] / 1000,
                'avg_us': s['total_us'] / s['count'] if s['count'] > 0 else 0,
                'pct': 100 * s['total_us'] / total_time if total_time > 0 else 0,
            })
        
        sort_key = {
            'total': lambda x: -x['total_ms'],
            'avg': lambda x: -x['avg_us'],
            'count': lambda x: -x['count'],
        }[sort_by]
        
        return sorted(results, key=sort_key)[:n]
    
    def breakdown(self) -> dict[str, float]:
        """Categorize GPU time by kernel type."""
        self._index_events()
        
        categories = {
            'NCCL AllReduce': 0,
            'NCCL AllGather': 0,
            'NCCL Other': 0,
            'MoE Kernels': 0,
            'Attention (Flash)': 0,
            'GEMM/Linear': 0,
            'Quantization': 0,
            'Elementwise/Activation': 0,
            'Triton Other': 0,
            'Other': 0
        }
        
        for k in self._kernels:
            name = k['name']
            dur = k.get('dur', 0)
            
            if 'nccl' in name.lower():
                if 'allreduce' in name.lower():
                    categories['NCCL AllReduce'] += dur
                elif 'allgather' in name.lower():
                    categories['NCCL AllGather'] += dur
                else:
                    categories['NCCL Other'] += dur
            elif 'fused_moe' in name or 'moe_align' in name or 'topk' in name.lower() or 'count_and_sort_expert' in name:
                categories['MoE Kernels'] += dur
            elif 'flash' in name.lower() or 'reshape_and_cache' in name:
                categories['Attention (Flash)'] += dur
            elif 'gemm' in name.lower() or 'cutlass' in name.lower() or 'cublas' in name.lower() or '_w8a8' in name:
                categories['GEMM/Linear'] += dur
            elif 'quant' in name.lower():
                categories['Quantization'] += dur
            elif 'silu' in name or 'act_and_mul' in name or 'elementwise' in name or 'reduce_kernel' in name:
                categories['Elementwise/Activation'] += dur
            elif 'triton' in name.lower():
                categories['Triton Other'] += dur
            else:
                categories['Other'] += dur
        
        # Convert to ms
        return {k: v / 1000 for k, v in categories.items()}
    
    def small_kernels(self, threshold_us: float = 10) -> list[dict]:
        """Find small kernels that may be fusion candidates."""
        self._index_events()
        
        stats = defaultdict(lambda: {'count': 0, 'total_us': 0})
        for k in self._kernels:
            dur = k.get('dur', 0)
            if dur < threshold_us:
                name = k['name']
                stats[name]['count'] += 1
                stats[name]['total_us'] += dur
        
        results = []
        for name, s in stats.items():
            results.append({
                'name': name,
                'count': s['count'],
                'avg_us': s['total_us'] / s['count'],
                'total_ms': s['total_us'] / 1000,
            })
        
        return sorted(results, key=lambda x: -x['count'])
    
    # -------------------------------------------------------------------------
    # Model Structure Inference
    # -------------------------------------------------------------------------
    
    def infer_structure(self) -> dict:
        """Infer model structure from kernel patterns."""
        self._index_events()
        
        kernel_counts = defaultdict(int)
        for k in self._kernels:
            kernel_counts[k['name']] += 1
        
        # Flash attention layers
        flash_combine = sum(v for k, v in kernel_counts.items() if 'flash_fwd_splitkv_combine' in k)
        flash_fwd = sum(v for k, v in kernel_counts.items() if 'flash_fwd' in k and 'combine' not in k and 'splitkv' not in k)
        
        # MoE layers
        moe_kernel = kernel_counts.get('fused_moe_kernel', 0)
        
        # AllReduce for TP degree estimation
        allreduce = sum(v for k, v in kernel_counts.items() if 'AllReduce' in k)
        
        dist = self.distributed_info()
        
        return {
            'attention_layers_estimate': max(flash_combine // 96, flash_fwd // 96) if flash_combine or flash_fwd else None,
            'moe_layers_estimate': moe_kernel // 96 if moe_kernel else None,
            'uses_flash_attention': flash_combine > 0 or flash_fwd > 0,
            'uses_splitkv_decode': flash_combine > 0,
            'uses_moe': moe_kernel > 0,
            'tp_degree': dist.get('world_size', 1),
            'allreduce_count': allreduce,
        }
    
    # -------------------------------------------------------------------------
    # Inductor Analysis
    # -------------------------------------------------------------------------
    
    def inductor_mapping(self) -> dict[str, list[str]]:
        """Map Inductor-generated files to kernels they invoke."""
        self._index_events()
        
        inductor_calls = [
            pf for pf in self._python_funcs 
            if '/tmp/torchinductor_' in pf.get('name', '')
        ]
        
        file_to_kernels = defaultdict(set)
        
        for pf in inductor_calls:
            file_name = pf['name'].split('(')[0].split('/')[-1]
            ts_start = pf['ts']
            ts_end = ts_start + pf.get('dur', 0)
            
            # Find CPU ops (including triton kernel launches) in this window
            for op in self._cpu_ops:
                op_ts = op['ts']
                if ts_start <= op_ts <= ts_end:
                    name = op['name']
                    if 'triton' in name.lower() or name.startswith('_w8a8'):
                        file_to_kernels[file_name].add(name)
        
        return {k: list(v) for k, v in file_to_kernels.items()}
    
    def decode_fusion(self, kernel_name: str) -> list[str]:
        """Decode fused ops from Inductor kernel name."""
        if 'fused_' not in kernel_name:
            return [kernel_name]
        
        # Extract ops after 'fused_'
        parts = kernel_name.split('fused_')
        if len(parts) < 2:
            return [kernel_name]
        
        fused_part = parts[1]
        # Remove trailing _N (variant number)
        if fused_part and fused_part[-1].isdigit():
            fused_part = '_'.join(fused_part.split('_')[:-1])
        
        # Split on _ but keep _to_copy together
        ops = []
        current = []
        for part in fused_part.split('_'):
            if part == 'to' and current and current[-1] == '':
                current.append(part)
            elif part == 'copy' and current and 'to' in current:
                current.append(part)
                ops.append('_'.join(current))
                current = []
            elif part:
                if current:
                    ops.append('_'.join(current))
                current = [part]
        if current:
            ops.append('_'.join(current))
        
        return ops
    
    def trace_kernel(self, kernel_name: str) -> dict:
        """Trace a kernel back to its Python source."""
        self._index_events()
        
        # Find the kernel
        kernel = None
        for k in self._kernels:
            if kernel_name in k['name']:
                kernel = k
                break
        
        if not kernel:
            return {'error': f'Kernel {kernel_name} not found'}
        
        result = {
            'kernel': kernel['name'],
            'duration_us': kernel.get('dur'),
            'grid': kernel.get('args', {}).get('grid'),
            'block': kernel.get('args', {}).get('block'),
        }
        
        # Find CUDA runtime call via correlation
        corr = kernel.get('args', {}).get('correlation')
        if corr:
            for rt in self._cuda_runtime:
                if rt.get('args', {}).get('correlation') == corr:
                    result['cuda_runtime'] = rt['name']
                    result['launch_ts'] = rt['ts']
                    
                    # Find active Python functions at launch time
                    rt_ts = rt['ts']
                    active_pfs = []
                    for pf in self._python_funcs:
                        pf_start = pf['ts']
                        pf_end = pf_start + pf.get('dur', 0)
                        if pf_start <= rt_ts <= pf_end:
                            name = pf.get('name', '')
                            if 'torchinductor' in name or 'vllm' in name.lower():
                                active_pfs.append({
                                    'name': name,
                                    'duration_us': pf.get('dur'),
                                })
                    
                    # Sort by duration (innermost = shortest)
                    active_pfs.sort(key=lambda x: x['duration_us'])
                    result['python_stack'] = active_pfs[:5]
                    break
        
        return result
    
    # -------------------------------------------------------------------------
    # CUDA Graph Analysis
    # -------------------------------------------------------------------------
    
    def uses_cuda_graphs(self) -> bool:
        """Check if trace uses CUDA Graphs."""
        self._index_events()
        return any('cudaGraphLaunch' in rt.get('name', '') for rt in self._cuda_runtime)
    
    def phases(self) -> list[dict]:
        """Get execution phases from user annotations."""
        self._index_events()
        
        phases = []
        for ua in self._user_annotations:
            phases.append({
                'name': ua['name'],
                'duration_ms': ua.get('dur', 0) / 1000,
                'ts': ua['ts'],
            })
        
        return sorted(phases, key=lambda x: -x['duration_ms'])


def main():
    parser = argparse.ArgumentParser(description='PyTorch Profiler Trace Analyzer')
    parser.add_argument('trace', help='Path to trace.json or trace.json.gz')
    parser.add_argument('--summary', action='store_true', help='Print executive summary')
    parser.add_argument('--top-kernels', type=int, metavar='N', help='Show top N kernels')
    parser.add_argument('--breakdown', action='store_true', help='Show time breakdown by category')
    parser.add_argument('--inductor', action='store_true', help='Show Inductor file mapping')
    parser.add_argument('--small-kernels', type=float, metavar='US', help='Find kernels smaller than US microseconds')
    parser.add_argument('--trace-kernel', type=str, metavar='NAME', help='Trace kernel back to source')
    parser.add_argument('--structure', action='store_true', help='Infer model structure')
    
    args = parser.parse_args()
    trace = PTTrace(args.trace)
    
    if args.summary:
        s = trace.summary()
        print(f"Device: {s['device']} x {s['num_gpus']}")
        if s['distributed']:
            d = s['distributed']
            print(f"Distributed: {d['backend']} rank {d['rank']}/{d['world_size']}")
        print(f"Total GPU Time: {s['total_gpu_time_ms']:.2f} ms")
        print(f"Kernel Launches: {s['total_kernel_launches']} ({s['unique_kernels']} unique)")
        print(f"CPU Ops: {s['total_cpu_ops']}")
    
    if args.top_kernels:
        print(f"\n{'%':>6} | {'Total (ms)':>10} | {'Count':>6} | {'Avg (μs)':>10} | Kernel")
        print("-" * 80)
        for k in trace.top_kernels(args.top_kernels):
            print(f"{k['pct']:5.1f}% | {k['total_ms']:10.1f} | {k['count']:6d} | {k['avg_us']:10.1f} | {k['name'][:40]}")
    
    if args.breakdown:
        print("\nTime Breakdown:")
        breakdown = trace.breakdown()
        total = sum(breakdown.values())
        for cat, ms in sorted(breakdown.items(), key=lambda x: -x[1]):
            pct = 100 * ms / total if total > 0 else 0
            print(f"  {pct:5.1f}% | {ms:8.1f} ms | {cat}")
    
    if args.inductor:
        print("\nInductor File → Kernel Mapping:")
        for file, kernels in trace.inductor_mapping().items():
            print(f"\n📁 {file}")
            for k in kernels:
                print(f"   🔷 {k[:70]}")
    
    if args.small_kernels:
        print(f"\nSmall Kernels (<{args.small_kernels} μs):")
        for k in trace.small_kernels(args.small_kernels)[:20]:
            print(f"  {k['count']:5d}x | avg {k['avg_us']:.1f} μs | {k['name'][:60]}")
    
    if args.trace_kernel:
        result = trace.trace_kernel(args.trace_kernel)
        print(f"\nKernel Trace: {result.get('kernel', 'Not found')}")
        if 'error' not in result:
            print(f"  Duration: {result.get('duration_us')} μs")
            print(f"  Grid: {result.get('grid')}, Block: {result.get('block')}")
            print(f"  Launch: {result.get('cuda_runtime')}")
            if result.get('python_stack'):
                print("  Python Stack (innermost first):")
                for pf in result['python_stack']:
                    print(f"    {pf['name'][:70]}")
    
    if args.structure:
        print("\nInferred Model Structure:")
        s = trace.infer_structure()
        for k, v in s.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
