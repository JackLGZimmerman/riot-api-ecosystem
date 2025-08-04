import time
import psutil
import functools
from typing import Callable, Any
from memory_profiler import memory_usage


def profile_resource_usage(fn: Callable[..., Any]) -> Callable[..., Any]:

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        proc = psutil.Process()
        proc.cpu_percent(None)

        start_time = time.perf_counter()
        mem_samples: list[float] = []
        try:
            mem_samples, result = memory_usage(
                proc=(fn, args, kwargs),
                retval=True,
                interval=0.1,
                timeout=None,
            )
        except Exception as exc:
            end_time = time.perf_counter()
            cpu_percent = proc.cpu_percent(None)
            delta_time = end_time - start_time

            if mem_samples:
                start_mem = mem_samples[0]
                peak_mem = max(mem_samples)
                delta_mem = peak_mem - start_mem
            else:
                start_mem = peak_mem = delta_mem = 0.0

            print(
                "Function %s raised %s; took %.4f s | CPU%%: %.1f | "
                "Mem start=%.2f MiB peak=%.2f MiB delta=%.2f MiB",
                fn.__name__,
                exc,
                delta_time,
                cpu_percent,
                start_mem,
                peak_mem,
                delta_mem,
            )
            raise

        end_time = time.perf_counter()
        cpu_percent = proc.cpu_percent(None)
        delta_time = end_time - start_time

        if mem_samples:
            start_mem = mem_samples[0]
            peak_mem = max(mem_samples)
            delta_mem = peak_mem - start_mem
        else:
            start_mem = peak_mem = delta_mem = 0.0

        print(
            "Function %s took %.4f s | CPU%%: %.1f | "
            "Mem start=%.2f MiB peak=%.2f MiB delta=%.2f MiB",
            fn.__name__,
            delta_time,
            cpu_percent,
            start_mem,
            peak_mem,
            delta_mem,
        )

        return result

    return wrapper
