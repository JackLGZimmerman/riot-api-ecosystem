# Hyper-parameter Testing Proceedure

Run isolated 5-epoch training jobs on the current token model.
Each trial uses a fresh Python process, seed `42`, BF16 AMP, diagnostics off,
and final test evaluation from that trial's best validation checkpoint.
