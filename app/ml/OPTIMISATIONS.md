# ML Optimisations

## Batch Size And Gradient Accumulation

Ran isolated 5-epoch training jobs on the current 10-interaction-token model.
Each trial used a fresh Python process, seed `42`, BF16 AMP, diagnostics off,
and final test evaluation from that trial's best validation checkpoint.

| physical batch | grad accum | effective batch | samples/s | test AUC | test acc | test loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 4096 | 36,995 | 0.5934 | 0.5687 | 0.6794 |
| 8192 | 1 | 8192 | 40,501 | 0.5929 | 0.5670 | 0.6800 |
| 10240 | 1 | 10240 | 41,044 | 0.5927 | 0.5666 | 0.6803 |
| 4096 | 2 | 8192 | 40,363 | 0.5931 | 0.5664 | 0.6802 |
| 5120 | 2 | 10240 | 40,645 | 0.5928 | 0.5664 | 0.6803 |
| 8192 | 2 | 16384 | 40,963 | 0.5881 | 0.5646 | 0.6813 |

Summary: gradient accumulation did not provide a useful win in this pass. At
matched effective batch sizes it was throughput-neutral to slightly slower and
accuracy-neutral to slightly worse. Increasing effective batch to `16384` with
accumulation hurt AUC over 5 epochs. The best throughput came from the largest
plain physical batch tested, `10240 x 1`; the best short-run accuracy came from
`4096 x 1`, but with about 10% lower throughput.

Note: trials must be isolated by process for clean throughput numbers. A first
attempt that reused one Python process across trials polluted the CUDA allocator
between runs and produced misleading allocator-cliff behavior.
