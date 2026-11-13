# Author implementations

FCM and CATCH comparisons use the authors' modules at fixed commits, with the compatibility changes recorded alongside each source specification:

| Method | Author repository | Recorded commit | Changes |
| --- | --- | --- | --- |
| FCM | https://github.com/mala-lab/FCM | `67e4c841098a0b4e161b9dfd3b0a5aefb34ec0e8` | [Patch](FCM/compatibility.patch), [source specification](FCM/SOURCE.json) |
| CATCH | https://github.com/decisionintelligence/CATCH | `3647c69be5eb56649b072596cf89098e689e20c3` | [Patch](CATCH/compatibility.patch), [source specification](CATCH/SOURCE.json) |

Run `python scripts/fetch_author_code.py` to download these recorded files from their authors and apply the patches. Downloaded source files remain outside this repository's tracked files. The upstream repositories supply their own usage terms; this repository does not assign a new license to the authors' code.

The event-warning endpoint adapters and evaluation drivers are in `src/vendor_fcm.py`, `src/vendor_catch.py`, and `scripts/`. Chronos-2 is installed via `chronos-forecasting` and downloaded by `scripts/prepare_hne_routed.py`; the [upstream implementation](https://github.com/amazon-science/chronos-forecasting) uses the Apache-2.0 license. Literature sources and citations are retained in `literature/`; third-party full-text papers are not mirrored here.
