# fixtures/

Small, hand-crafted, fictional sample data - NOT real challenge data - matching the
official schema. Use these to develop and unit-test the pipeline without needing the
full dataset in context.

`train/` includes a matching ground truth and deliberately covers: legal-suffix and
punctuation variation (S1-0001), clean abbreviation differences (S1-0002), a
landmark-based address (S1-0003), a name variant that adds a token ("and Cafe",
S1-0004), and one deliberate HARD NEGATIVE (S3-0003 "City Bakery" is a similar but
distinct nearby business at 47 Oak Ave, NOT a true match to S1-0004 at 45 Oak Ave -
useful for testing that the model doesn't reflexively merge chain/branch look-alikes),
plus S3-0004 as a second near-miss (Acme Hardware at 501 vs 500 Main St).

`test/` covers: an unseen-in-train country (France, S1-1001), a multi-match entity
(S1-1002 plausibly matches BOTH S2-1002 and S3-1002), and a deliberate singleton with
no true match in either source (S1-1004) to sanity-check singleton handling end-to-end.

No ground truth is provided for `test/`, matching the real challenge format - use it to
exercise the inference path, not to compute F0.5 (use the train fixture, or your real
validation split, for that).
