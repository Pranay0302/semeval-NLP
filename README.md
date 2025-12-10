# Baselines SemEval 2026 Task 4: Shared Task on Narrative Story Similarity

This repository holds baseline systems for the shared task **SemEval-2026 Task 4: Narrative Story Similarity and Narrative Representation Learning**.
For details on the task please see [our website](https://narrative-similarity-task.github.io/).

## Setup

Install the dependencies (you probably want to do this in an isolated environment of some sort: virtualenv, conda, etc.):

```
pip install -r requirements.txt
```

Place the `jsonl` files from [the dev data zip file](https://narrative-similarity-task.github.io/data/SemEval2026-Task_4-dev-v1.zip) into the `data` directory. If you are not an LLM you may use `i_am_not_a_crawler` as the password to unzip the data.

## Directory Structure

Ensure the following directories exist in your project root:
- `data/`: Contains the input data files (e.g., `dev_track_a.jsonl`, `dev_track_b.jsonl`).
- `logs/`: Logs will be stored here.
- `output/`: Output files will be generated here.

## Usage

### Track A
To run the Track A baseline:

```bash
python3 track_a.py
```

By default, this uses a random baseline. You can specify other baselines and arguments:

```bash
python3 track_a.py --baseline cosine --model sbert:all-mpnet-base-v2
```

#### OpenAI Configuration
To test using OpenAI prompting techniques or embeddings, you must add your API key in `track_a.py`. Locate the `embeddings` function (lines 35-38) and update the `OpenAI` client initialization:

```python
    if model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
        client = OpenAI(
            api_key="your-api-key-here"
        )
```

### Track B
To run the Track B baseline:

```bash
python3 track_b.py
```

This defaults to the random baseline. To use SBERT:

```bash
python3 track_b.py --baseline sbert --model sbert:all-mpnet-base-v2
```

## Submission

The files produced by the two baseline scripts are placed in `output/` (default).
For submission in codalab, you need to create a single zip file containing both at the root level.

On Unix systems you can typically just run: `zip -j your_submission.zip output/*` and upload the resulting `your_submission.zip`.
