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