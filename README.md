# ReMedQA benchmark

Measure strict answer accuracy across ReMedQA's six closed-answer formats
with an OpenAI-compatible Chat Completions API.

Requires Python 3.10+ and Git. No Python packages are required. Run the
commands below from this directory.

## Setup

Get the fixed dataset and answer parser:

```sh
git clone https://github.com/disi-unibo-nlp/remedqa.git remedqa
git -C remedqa checkout 2caa66392f5d7c9632d8c24d6a9391df79c6555a
```

## Run

Supply the endpoint and model. Full coverage is the default:
3,154 questions × six formats = 18,924 requests. API calls use your credit.
Use `plan` to check settings without API calls. For example:

```sh
export MOONCO_API_KEY="your-api-key"
python3 remedqa_eval.py plan \
  --base-url https://api.moonco.one/v1 --model moon-1 --out results/moon-1
python3 remedqa_eval.py run \
  --base-url https://api.moonco.one/v1 --model moon-1 --out results/moon-1
```

To compare another model, use its endpoint and key:

```sh
export OTHER_API_KEY="your-other-api-key"
python3 remedqa_eval.py run \
  --base-url https://your-api.example/v1 \
  --model your-model --api-key-env OTHER_API_KEY --out results/other-model
```

The endpoint must support `POST /chat/completions`. Repeat a command to
resume; saved responses are not requested again. Use a new output directory
for different settings, and only one process per directory. Use the same
evaluator version to resume or score a run.

Defaults: 8,192 output tokens, four calls in flight, 30 request starts/minute.

## Run a fixed question set

Use the same percentage and seed for each model:

```sh
python3 remedqa_eval.py run \
  --base-url https://api.moonco.one/v1 --model moon-1 \
  --percent 20 --seed 20260723 --out results/moon-1-20
```

This selects 633 questions and makes 3,798 requests. Other percentages use
`--percent`. Selection is fixed within each task, rounded up, and keeps all
six formats for each question. Full coverage is the default.

## Results

Read `report.md` in the output directory. The manifest records settings and
question identities; responses and strict verdicts are also saved. Outputs
under `results/` are ignored by Git. Keep custom output paths out of commits.

To score saved answers without an API key or model calls:

```sh
python3 remedqa_eval.py score --out results/moon-1
```

Strict ClosedAvg averages MedQA, MedMCQA, and the mean of six MMLU subjects
within each format, then averages the six formats. The fixed upstream parser
checks final answers. The 95% interval uses 10,000 bootstrap draws with seed
42 and keeps each question's six answers together. Incomplete runs have no
benchmark score. Fresh runs can vary; accuracy does not establish clinical
safety.

## Licence

[MIT](LICENSE) covers this repository's code and documentation. It does not
cover ReMedQA data, prompts, parser, or source datasets. These are loaded
from the separate checkout. See the [official ReMedQA repository](https://github.com/disi-unibo-nlp/remedqa)
for their licence and attribution information.
