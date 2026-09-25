# Normalization dictionaries

`suffixes.json` and `abbreviations.json` are flat `{"observed form": "canonical form"}`
maps loaded by `src/business_entity_resolution/normalization/normalize.py`.

These MUST be mined from the provided train/test corpus (frequency analysis of name
tails for suffixes.json, of address tokens for abbreviations.json) - never hand-authored
from general knowledge and never sourced externally. See docs/architecture.md §2-§3 and
CLAUDE.md §2/§7.

Example shape:
```json
{
  "pvt ltd": "private limited",
  "pvt": "private",
  "ltd": "limited",
  "corp": "corporation"
}
```
Keep entries lowercase (normalization lowercases before applying the map). Multi-word
keys are matched before single-word keys (longest-key-first), so "pvt ltd" won't be
partially clobbered by a standalone "ltd" rule.

Currently empty placeholders - populate these as the first real output of EDA (§2)
before blocking/features are expected to perform well.
