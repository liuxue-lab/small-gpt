# Day 3 Streaming Corpus Pipeline Design

## 1. Objective

Build a reproducible and resumable streaming pipeline for collecting a bounded
FineWeb-Edu corpus without saving raw full-corpus shards.

The pipeline must clean, deduplicate, split, shard, verify, and record the
processed corpus before tokenizer training begins.

## 2. Fixed Dataset Identity

- Dataset: `HuggingFaceFW/fineweb-edu`
- Configuration: `sample-10BT`
- Split: `train`
- Revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- Access mode: streaming
- Collection unit: provided GPT-2 tokens

The provided token count is a collection budget. It is not the final token
count produced by the project tokenizer.

## 3. Fixed Collection Profiles

| Profile | Target provided tokens | Shard target | Estimated shard groups |
| --- | ---: | ---: | ---: |
| Pilot | 2,000,000 | 500,000 | 4 |
| Full | 350,000,000 | 5,000,000 | 70 |

A document is never divided between shards. A shard may therefore exceed its
target by the token count of its final document.

## 4. Output Layout

```text
data/processed/fineweb_edu_corpus/
├── manifest.json
├── state.json
└── shards/
    ├── shard-00000/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   ├── test.jsonl
    │   └── metadata.json
    └── shard-00001/
        ├── train.jsonl
        ├── validation.jsonl
        ├── test.jsonl
        └── metadata.json
```

Each shard directory represents one global provided-token interval. Records
inside that interval are written to separate train, validation, and test JSONL
files.

## 5. Record Processing

For every streamed source record:

1. normalize Unicode and whitespace;
2. reject missing, empty, short, non-English, or low-quality text;
3. calculate the SHA-256 digest of normalized text;
4. reject an exact duplicate already present in a completed or current shard;
5. assign the split deterministically from the text digest and seed;
6. write only the cleaned record to its split JSONL file;
7. add its provided token count to the collection and shard totals.

Formal collection does not save raw source records.

## 6. Atomic Shard Finalization

A shard is first written to a temporary directory:

```text
shards/.shard-00000.tmp/
```

After its JSONL files are flushed:

1. calculate each file size and SHA-256;
2. write `metadata.json`;
3. atomically rename the temporary directory to `shard-00000`;
4. atomically update `manifest.json` and `state.json`.

An incomplete temporary shard is never treated as completed output.

## 7. Resume Strategy

When resuming:

1. remove or ignore incomplete `.tmp` shard directories;
2. discover completed shard directories;
3. verify their metadata, file sizes, and SHA-256 values;
4. rebuild the exact-text hash set from completed JSONL records;
5. recover the last committed source-record position;
6. reopen the fixed dataset revision;
7. skip committed source records and continue with the next shard.

Completed shard directories are the durable source of truth. The state and
manifest files are validated against them and can be rebuilt if stale.

## 8. Required Invariants

The implementation and tests must verify:

- every completed JSONL file can be parsed line by line;
- every cleaned record belongs to exactly one split;
- `text_sha256` is globally unique across all shards and splits;
- split assignment is deterministic;
- statistics satisfy input, removal, retention, split, and token conservation;
- manifest file sizes and SHA-256 values match the files on disk;
- a controlled interruption can resume without loss or duplication;
- rerunning a completed profile is idempotent;
- generated corpus files remain ignored by Git.

## 9. Planned Command

```powershell
python scripts/build_fineweb_edu_corpus.py `
    --config configs/data_fineweb_edu.yaml `
    --profile pilot
```

A small smoke run must pass before the 2M-token Pilot is started.