# Optimization source archive

The adopted production path is documented in [GPU execution](../gpu_efficiency.md).
Historical experiment reports and plans in this directory retain the rationale,
measured results, rejected candidates and numerical limits; their commands target
the old checkout, not the current production CLI.

`optimization_sources_20261019.zip` is a byte-for-byte copy of
`reports/cpu_prefetch_20261019/source_snapshot.zip`, captured after the final
300-update P acceptance. It includes all 110 source/config/test/doc paths and
the experiment scripts, including the retired A/C/P trainer, Mask, bool-fill,
rotation-only/compiler diagnostics and their regression tests.

SHA256: `c2ccaec0f5ae4232934795a09b9520f017f4471700412439cb2730e3f5178936`.

Extract into a separate empty directory to inspect/reproduce old source bindings:

```bash
unzip docs/archive/optimization_sources_20261019.zip -d /tmp/diffphys-experiments
```

Existing runs, source-bound checkpoints, Nsight traces and audit capsules remain
under the original local `reports/`, `runs/`, `checkpoints/` paths. They are not
bundled into the source archive, overwritten, or removed. Use their original
source snapshot for exact historical resume; never modify checkpoint hashes.

Retired active Python paths are listed in
`reports/production_promotion_20261019/removed_paths.json`; the immediate
pre-cleanup snapshot is `reports/production_promotion_20261019/before_cleanup.zip`.
