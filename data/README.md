# Data

## Selected Dataset

| Item | Value |
|---|---|
| Dataset | HuggingFaceFW/fineweb-edu |
| Configuration | sample-10BT |
| Split | train |
| Language | English |
| Approximate size | 10 billion GPT-2 tokens |
| Access method | Streaming |
| License | Open Data Commons Attribution License (ODC-By) v1.0 |
| Source | Common Crawl |

Dataset card:

<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>

License:

<https://opendatacommons.org/licenses/by/1-0/>

Common Crawl terms:

<https://commoncrawl.org/terms-of-use>

## Why This Dataset

FineWeb-Edu contains English web pages selected using an educational-quality
classifier. The `sample-10BT` configuration is large enough for the planned
300M–500M token pretraining run while still supporting streaming access.

The complete dataset will not be downloaded. Scripts will stream only the
documents needed for inspection, preprocessing, tokenizer training, and model
pretraining.

## Expected Fields

| Field | Description |
|---|---|
| `text` | Extracted web-page text |
| `id` | Document identifier |
| `dump` | Common Crawl snapshot |
| `url` | Original page URL |
| `date` | Crawl date |
| `file_path` | Source file path |
| `language` | Detected language |
| `language_score` | Language confidence score |
| `token_count` | Approximate GPT-2 token count |
| `score` | Educational-quality score |
| `int_score` | Rounded educational-quality score |

## Local Directory Policy

Generated data is stored under:

- `data/raw/`: downloaded or streamed raw samples;
- `data/processed/`: cleaned and split text;
- `data/tokenized/`: binary token data generated after tokenizer training.

Raw, processed, and tokenized data must not be committed to Git. Only scripts,
documentation, tests, and very small synthetic test fixtures may be committed.

## Processing Rules

1. Read the dataset in streaming mode.
2. Validate required fields before processing.
3. Normalize Unicode and whitespace conservatively.
4. Remove empty and abnormally short documents.
5. Retain English documents that pass the language-confidence threshold.
6. Perform exact deduplication after normalization.
7. Split documents deterministically into train, validation, and test sets.
8. Use a fixed seed of `42`.
9. Use only the training split to train the tokenizer.
10. Record document counts before and after every filtering step.

The planned document-level split is:

- train: 98%;
- validation: 1%;
- test: 1%.

## Known Risks

- Web data may still contain harmful content, factual errors, bias, or personal information.
- Exact deduplication does not remove all near-duplicate documents.
- Source pages may be subject to additional terms or copyright restrictions.
- FineWeb-Edu contains relatively little source code.
- The provided `token_count` uses the GPT-2 tokenizer and is only an estimate
  for this project's custom BPE tokenizer.