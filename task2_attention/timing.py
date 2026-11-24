import statistics
import time

import torch


def sync():
    """Wait for queued gpu work so timers measure finished work"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn,warmup,reps):
    """Run a function after warmup calls and return the mean and spread of its wall time in milliseconds"""
    times = []
    for r in range(warmup+reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        if r >= warmup:
            times.append(1000*(time.perf_counter()-start))
    return statistics.mean(times),statistics.stdev(times) if len(times) > 1 else 0.0


def peak(fn):
    """Peak gpu memory in gigabytes that one call adds above what was already allocated"""
    if not torch.cuda.is_available():
        return None,None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    sync()
    return (torch.cuda.max_memory_allocated()-base)/1e9,torch.cuda.max_memory_reserved()/1e9
