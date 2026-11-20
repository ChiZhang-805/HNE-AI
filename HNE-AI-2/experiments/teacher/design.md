# Conditional-information experiment

This first execution is a source-fold development study, not completion of the
five-fold, three-seed paper design. It compares the preserved shared predictor,
a separately trained H-only predictor, and frozen H plus switchable N/E residuals.

The source is BrisT1D-Open, outer fold 0, seed 20261114. Reference observations are
loaded without event labels or label-derived origin eligibility. Twelve source
training people form three excluded scoring groups of four. For each component,
the other eight people are split into six fitting and two selection people.
Epoch, learning-rate and architecture choices use those selection people's HNE
forecast NLL only. Chosen budgets are then refitted on the permissible eight;
the excluded four never enter fitting or selection. Four separate source
validation people are audited only after selection.

Three fixed learning rates, at most 24 epochs and patience 5 are used for each
architecture. The residual starts from the inner-selected H checkpoint during
selection, and from the refitted H checkpoint during refitting. Its epoch-zero
candidate is retained. Its history branch is frozen. The correction subtracts
the same head evaluated with a zero source message, so disabled/absent messages
cannot add a generic H-only output bias. This is not a claim of equal parameter
counts: H has 120,736 stored parameters; residual fitting adds 32,960 trainable
parameters. Fit, selection and refit costs are recorded separately.

The design follows the previously audited separation of shared and conditional
parts and explicit identity when intervention is disabled. It does not reproduce
GenARM/PARM language-task algorithms or inherit their empirical conclusions.
Their original source snapshots and licenses stay in the literature directory.

Evaluation reports H/HN/HNE losses, coverage and PIT before and after personal
calibration. Personal fit uses days 28–42; reference scoring uses days 42–56.
A fixed monitoring window scores days 74–88 when present, selected without event
labels. A seven-patch older-source condition preserves chronological legality
and tests reliance on aligned information. It does not by itself prove causal
relationships or completely rule out every extra-capacity explanation.

Paired person bootstraps describe uncertainty conditional on the fitted models;
their overlapping training pools and model-selection uncertainty are disclosed.
All negative results and epoch-zero choices are retained. The selected component
bundle can later supply the same evidence to the internal and fusion comparisons.
