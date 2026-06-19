# GAUGE Publication Bundle

## Overview
This bundle is organized for publication-style reproducibility while preserving the original repository untouched.

- Original repo untouched: `/mnt/raid5/xujing/KG`
- Bundle root: `KG_publication/GAUGE_publication_bundle`
- `src/GAUGE`: full source copy (no file removed)
- `targets/`: reproducibility targets for selected result chains
- `snapshots/`: relative symlinks to existing heavy result directories
- `runtime/`: one-command runners and smoke checks

## Environment
Default environment assumptions are kept compatible with the original setup.
You can override roots via env vars in `runtime/env.sh`:
- `KGPUB_KG_ROOT`
- `KGPUB_DATA_ROOT`
- `KGPUB_PY_ROOT`
- `KGPUB_PRISM_PATCH_ROOT`

## Main Entrypoints
- `runtime/run_01_random_cell_split.sh`
- `runtime/run_02_drug_split.sh`
- `runtime/run_07_tcga_task01.sh`
- `runtime/run_publication_v7_step1_2_3.sh`
- `runtime/run_cnm_core.sh`
- `runtime/run_cnm_noveldrug.sh`
- `runtime/run_peru_abc.sh`

## Validation
Run:

```bash
bash runtime/smoke_check.sh
```

The smoke check validates importability, key links, and core entry command startup.
