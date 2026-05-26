# Precision-First Thresholding

The first live rollout is shadow mode, so thresholds should be tuned to protect user
trust before any blocking is enabled.

## Default Policy

- Optimize high precision first.
- Review false positives by language before raising enforcement.
- Keep `safe` comments unblocked during shadow mode.
- Log every prediction with model version, confidence, latency, and final product action.

## Suggested Gates

- Shadow mode: collect predictions and human review samples only.
- Limited enforcement candidate: require high precision per enforced label on internal OTT data.
- Full enforcement candidate: require stable precision across Indonesian, English, and mixed-language chat.

Public bootstrap datasets are useful for pipeline testing, but they are not enough to approve
blocking in production. Use real OTT chat and human labels for threshold selection.
