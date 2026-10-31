# Third-party inference source

Kronos upstream: https://github.com/shiyu-coder/Kronos

A minimal inference-only source snapshot from the supplied project archive is bundled under
`finvex/analysis/kronos/_vendor/`. Its MIT license is preserved there in `LICENSE`; provenance and
original file hashes are in `UPSTREAM.json`. The exact upstream commit is unknown. Only the absolute
internal import in `model/kronos.py` was changed to a relative one.

Model weights are not redistributed. Check each selected Hugging Face model card and license before
redistribution or deployment. The upstream license does not define the license of Finvex itself.
