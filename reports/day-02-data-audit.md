# Day 2 Data Audit

## 1. Audit Objective

This audit verifies the source, license, schema, sample quality, storage
requirements, and initial cleaning policy for the Small GPT pretraining corpus.

No full dataset was downloaded during this audit.

## 2. Selected Dataset

| Item | Value |
|---|---|
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Configuration | `sample-10BT` |
| Split | `train` |
| Language | English |
| Approximate size | 10 billion GPT-2 tokens |
| Access method | Streaming |
| Fixed revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Source | Common Crawl |

Dataset card:

<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>

## 3. License and Usage Conditions

FineWeb-Edu is released under the Open Data Commons Attribution License
(ODC-By) v1.0. Its use is also subject to the Common Crawl Terms of Use.

- ODC-By 1.0: <https://opendatacommons.org/licenses/by/1-0/>
- Common Crawl terms: <https://commoncrawl.org/terms-of-use>

This project retains dataset attribution and source metadata for traceability.
The corpus is used for research and educational model pretraining.

The dataset license does not remove every possible restriction associated with
individual source pages. Generated models and public releases must therefore
include appropriate dataset attribution and a description of known risks.

## 4. Reproducible Sampling Procedure

The sample was obtained with:

- `datasets==5.0.1`;
- `streaming=True`;
- a fixed dataset revision;
- configuration `sample-10BT`;
- the first 1,000 streamed records;
- no preprocessing before inspection.

The sample is intended for pipeline validation and preliminary auditing. It
must not be treated as an unbiased statistical estimate of the complete corpus.

The raw sample is stored locally at:

```text
data/raw/fineweb_edu_sample.jsonl
```
## 5. Sample Audit Results

The following results were produced from the first 1,000 streamed records and
saved in `reports/day-02-inspection.json`.

### 5.1 Record Integrity

| Metric | Result |
| --- | ---: |
| Records requested | 1,000 |
| Records inspected | 1,000 |
| Empty texts | 0 |
| Exact duplicate texts | 0 |
| Duplicate IDs | 0 |
| English records | 1,000 |
| Records missing `date` | 1,000 |

### 5.2 Character and Token Statistics

| Metric | Minimum | Maximum | Mean | Median |
| --- | ---: | ---: | ---: | ---: |
| Characters per record | 307 | 124,276 | 4,893.27 | 2,811.5 |
| Provided GPT-2 tokens per record | 80 | 26,697 | 1,057.56 | 615.5 |

The total provided GPT-2 token count of the inspected sample is 1,057,560.
These are dataset-provided GPT-2 token counts, not token counts from the
tokenizer that will later be trained for this project.

### 5.3 Language and Educational Scores

| Metric | Result |
| --- | ---: |
| Minimum language score | 0.665794 |
| Maximum language score | 0.994643 |
| Mean language score | 0.94 |
| Median language score | 0.945322 |

| Educational score | Records |
| ---: | ---: |
| 3 | 856 |
| 4 | 142 |
| 5 | 2 |

## 6. Day 2 Pipeline Validation

The Day 2 data pipeline covers:

- text normalization and record filtering;
- SHA-256 exact-text deduplication;
- deterministic train, validation, and test assignment;
- atomic output writing.

The resulting files contain:

| File | Records |
| --- | ---: |
| `data/raw/fineweb_edu_sample.jsonl` | 1,000 |
| `data/processed/train.jsonl` | 985 |
| `data/processed/validation.jsonl` | 7 |
| `data/processed/test.jsonl` | 8 |

The processed split counts sum to 1,000, matching the raw sample count.

## 7. Limitations and Decisions

- The audit covers only the first 1,000 streamed records and is not a
  representative statistical analysis of the complete dataset.
- All inspected records are missing the `date` field, so the pipeline must not
  depend on that field.
- No exact text duplicates were found in this sample. This does not imply that
  the complete dataset contains no duplicates.
- Dataset-provided GPT-2 token counts are suitable for initial budgeting, but
  the final token count must be measured with the project tokenizer.
- The fixed dataset revision must remain part of every reproducible extraction
  command and manifest.

## 8. Conclusion

The inspected FineWeb-Edu sample is suitable for controlled pipeline
development and a bounded pilot run. It is not sufficient evidence for
starting an unrestricted full-corpus download.

Before scaling the data extraction, Day 3 must add bounded token targets,
sharded output files, manifests, checksums, and resumable processing.