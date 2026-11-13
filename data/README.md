# Data sources and attribution

The public experiment records contain protocol-derived future-event labels, model-generated scores, relative forecast days, fold assignments, and the identifiers already used in the two public releases. They are derived experimental outputs, not clinical event annotations. HNE's future-event definition is in `configs/daily_protocol.json`.

| Dataset | Authors and release | License | Local preparation |
| --- | --- | --- | --- |
| RESILIENT | Nilforooshan, R., & Barnaghi, P. (2025). *The RESILIENT Dataset: Multimodal Monitoring of Ageing-Related Comorbidities and Cognitive Decline* (v3). [Zenodo](https://doi.org/10.5281/zenodo.16755408). Related descriptor: Céspedes Gómez et al. (2025), [Scientific Data](https://doi.org/10.1038/s41597-025-05958-x). | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Download `Sleepmat_Watch_Data.zip`. |
| BrisT1D-Open | Gordon James, S., Armstrong, M., O'Kane, A., Emerson, H., & Abdallah, Z. (2025). *BrisT1D-Open Dataset*. [University of Bristol](https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08). | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Download the public archive containing `device_data/processed_state/P*.csv`. |

The two licenses were checked against the authors' repository records. Derived labels/predictions are attributed to these sources and distributed under CC BY 4.0. Changes comprise hourly/day aggregation, personal-reference normalization, future-event construction, participant-separated splitting, and model prediction; the exact implementation is included in this repository. Citation does not imply endorsement by the dataset authors.

After downloading the two archives, run `scripts/prepare_public_main.py` as shown in the project README. Prepared inputs, pretrained model weights, and cached representations are generated locally. The fixed Chronos-2 model revision is recorded in `configs/hne_routed.json`.

Other cohorts appear as aggregate results and paper tables. Their references and preparation scripts are listed in `literature/references.bib` and `scripts/`. Download each cohort from its own release before rerunning those experiments.
