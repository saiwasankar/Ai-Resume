# Labeled matcher data

RoleFit can train a supervised evidence matcher from a JSONL file containing
human-reviewed pairs. Do not add private resumes, protected attributes, or
unlicensed data to this repository.

Each line must contain:

```json
{"job_id":"backend-001","requirement":"Build Python APIs","evidence":"Built Python APIs with FastAPI for 3 years","label":1}
```

`label` is `1` when the resume passage is valid evidence for the requirement and
`0` when it is not. `job_id` is required so evaluation can hold out complete
job postings rather than leaking near-duplicate requirements across splits.

For an industry benchmark, collect consented or licensed resumes, have at
least two qualified reviewers label each pair, adjudicate disagreements, and
keep a final test set untouched by model development. The example above is a
schema example, not a benchmark.
