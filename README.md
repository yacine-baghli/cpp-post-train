# cpp-post-train

Post-training **Qwen3.5-9B** on [Tinker](https://tinker-docs.thinkingmachines.ai/) to solve hard competitive
programming problems in C++20, for the "Post-Train with Tinker" track of the Vals AI hackathon
(November 14–15, 2026).

This README explains what the task is, what we built, what we tried, and what actually happened,
including the many attempts that made the model worse before one made it better.

## The task

- Submit a checkpoint of a model with at most 9B parameters, post-trained from a base model on Tinker.
- The grader gives the model one problem at a time and two tools: one compiles and runs C++ on any
  input, the other submits a solution and reports the result on the sample tests.
- Up to 50 submissions per problem. The **last submission that compiles** is graded on hidden tests.
  The model ends the task by answering `EXIT`.
- Small hidden tests are worth 40 points and large ones 60. Ties are broken by total turn count.
- We cannot set the grader's prompt: the only levers are the weights and the sampling settings.

## Result

| Version | Leaderboard score | Duration per test |
|---|---|---|
| `cyan2` (expert iteration on the model's own runs) | **45.81** | 9m08s |
| `lagon` (same, plus brute-force fallback examples) | 41.20 | 5m10s |
| `ciel` (untrained reference) | 32.57 | 12m51s |
| `outremer` (earlier self-distillation attempt) | 28.17 | 2m50s |

`cyan2` put the team second of 17, 0.52 points behind first place (46.33) when it was graded. It is
sampled at **temperature 1.0**, not the 0.6 that suits the base model; that setting turned out to
matter as much as the training.

Each score above is a single grading. Other leaderboard results are not recorded here.

## What worked

**Expert iteration on the base model's own behaviour.**

1. Run the untrained model on public Codeforces problems inside a copy of the grader's tool loop,
   several times per problem (6,870 runs over about 1,700 problems).
2. Score every run the way the grader does: 40 points if the final program passes all small tests,
   60 more if it passes the large ones.
3. Keep the best run per problem, cut it at the first submission of the winning program, append `EXIT`.
4. Train on those runs, reasoning included (1,165 runs, 216 of them on problems rated 1600–2800).
5. Sample at temperature 1.0.

That is `cyan2`. Two findings made it work after a first round that looked like a failure:

- **Self-training sharpens the model, and it starts to loop.** Trained on its own successful runs and
  sampled at 0.6, the model ran its thinking to the output limit far more often than the base and
  scored 22 locally. The same checkpoint at temperature 1.0 was back at the base's level. Across the
  sweep on `lagon`, 1.0 beat both 0.9 and 1.2.
- **The local score ranked the models correctly but badly underestimated the gain.** Locally `cyan2`
  was 2 points above the base on the strict score and 5 with partial credit; on the leaderboard it
  was 13 points above. We do not know why the gap is so much larger on the hidden problems.

**The fallback examples did not help.** `lagon` adds 306 examples built from the model's own verified
brute-force solutions (when its thinking was cut off without a tool call, or two submissions have
failed, submit a brute force and stop). It scored 41.20, below `cyan2` without them. With one grading
each, part of that difference may be noise, but there is no evidence the fallback is worth having.

## Local scoring

The leaderboard allows 4 submissions per hour, one at a time, so we built a local stand-in.

- **Harness** (`harness.py`): the same loop as the hackathon describes. A sandbox compiles with
  `g++ -std=c++20 -O2`, the `run` tool executes code on a given input, the `submit` tool reports
  sample results, and the last compiling submission is graded.
- **Validation set**: 64 recent problems from the test split of
  [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces), kept separate from
  all training problems. The official tests play the role of "small" tests; the largest generated
  tests of each problem play the role of "large" tests. 52 problems have both.
- **Score**: 40 points if every small test passes plus 60 if every large test passes, averaged
  ("strict"). A second number gives partial credit per test.
- **Noise**: one pass over 52 problems is good to about ±6 points, so the key models were scored
  two or three times.

The grader's real tool names and prompt are unknown, so the harness draws from four tool-name and
system-prompt variants to keep a fine-tuned model from depending on one wording.

| Model (temperature) | Strict | With partial credit | Runs ending with EXIT |
|---|---|---|---|
| `lagune` (1.0), second training of the `lagon` recipe | 33.1 | 39.1 | 36 of 64 |
| `cyan2` (1.0), `lagon` data without the fallback examples | 32.7 | 39.9 | 33 of 64 |
| `lagon` (1.0), two passes | 31.0 | 38.0 | 89 of 128 |
| base (0.6), three passes | 30.4 | 34.9 | 30 of 128 |
| `ocean` (1.0), earlier version with less data | 30.0 | 36.2 | 40 of 64 |
| `lagon` (0.9) / (1.2) | 27.3 / lower than 1.0 | 33.6 / lower | |
| `lagune` scaled to 75% in weight space (1.0) | 27.7 | 37.6 | 30 of 64 |
| `cyan` (1.0) / (0.6), first expert-iteration round | 27.3 / 22.3 | 34.9 / 23.2 | |
| `glacier` (1.0), policy gradient | 22.3 | 28.4 | 28 of 64 |

## Everything we tried, in order

**1. Teacher distillation with a hidden strategy (`azur`).**
DeepSeek-V4.1-Flash solved training problems while following a routine given only to it: brute force
first, stress-test the efficient solution, then submit. The teacher solved about 98% of problems. The
9B trained on those trajectories tried to reproduce the teacher's long analysis, repeated itself to the
output limit and never submitted. About 13 locally.

**2. Short first turn (`cobalt`).** With the first turn shortened to brute-force reasoning only, the 9B
wrote brute forces without understanding the statement. About 13 locally.

**3. Actions only (`indigo`, later `nuit` and `abysse`).** Removing the loss on reasoning so the model
keeps its own thinking sounded safe and was not: it either stopped thinking, answered EXIT after failed
samples, or (on native runs) never said EXIT at all. 14 to 18 locally, 3 for `abysse`.

**4. The model's own first turn (`saphir`, `marine`, `outremer`).** The base 9B ignores a strategy in
the system prompt but follows one placed right after the problem statement. Sampling its own
brute-force or "best shot" first turns and training on the verified ones gave fast, tidy models that
stopped too early: `outremer` scored 28.17 on the leaderboard, below the base. The base's long
thinking and retries are worth points.

**5. Expert iteration (`cyan`, `ocean`, `lagon`, `lagune`, `cyan2`).** Described above. This is the
family that beat the base.

**6. Policy gradient (`glacier`).** Same runs, weighted by advantage: each run's score minus the mean
over the runs of the same problem, divided by their spread, positive weights for better-than-average
runs and smaller negative weights for worse ones. Locally it got worse as training went on (28.5 at
step 120, 20.8 at step 180, 22.3 at the end). With 1,695 runs it had a small fraction of the data that
published RL recipes use.

**7. Weight-space scaling.** Tinker would not accept uploaded weights for our organisation, so we
scaled an adapter without uploading anything: optimiser steps on a zero-weight loss have zero
gradient, so only weight decay moves the adapter, and each step multiplies both LoRA factors by
`1 - lr * wd`. Measured on the exported adapter, a 75% target gave 75.5% and a 50% target 50.5%.
The scaled models sat between the base and the full adapter and were not better than either end.

## What we learned

- Training a 9B model on a much stronger model's reasoning made it worse, long or short.
- Training it on its own successful runs works, provided the sampling temperature is raised to undo
  the sharpening.
- If a turn's reasoning is excluded from the loss, nothing tells the model to start that turn.
- Tidy behaviour is not the goal. The base model's messy long runs beat a decisive model by 4 points;
  what helped was keeping that behaviour and reinforcing the runs that succeed.
- A local metric can rank correctly and still misjudge the size of a gain by a wide margin.
- Sampling cost was badly underestimated twice. Multi-turn runs resend a long prompt at every turn,
  and prompt tokens, not generated tokens, dominated the bill.
- Token boundaries after `<think>\n` differ between a rendered conversation and a generated
  continuation when the reasoning is empty. `sft.py` handles this explicitly; before the fix some
  trajectories silently produced no training sequence.

## Repository layout

| File | Purpose |
|---|---|
| `harness.py` | Tool loop, C++ sandbox, grading, hidden strategy prompts |
| `problems.py` | Turns `open-r1/codeforces` rows into plain-text problems |
| `sft.py` | Renders trajectories with the model's chat template, builds loss masks and advantage weights, runs LoRA training |
| `modal_app.py` | Orchestration on Modal: data prep, sampling, grading, data-set builders, training, weight-space scaling |
| `inspect_traj.py` | Downloads a trajectory file and summarises behaviour |
| `serve.py` | vLLM server for the base model on Modal GPUs (written, not used: needs a payment method) |
| `smoke_test.py`, `common.py` | Account check and shared configuration |

Trajectories, logs and problem data are not in the repository.

## Reproducing `cyan2` and `lagon`

Requirements: [uv](https://docs.astral.sh/uv/), a Tinker API key and a Modal account. Data
preparation, C++ execution and orchestration run on Modal (CPU only); sampling and training run on
Tinker.

```bash
uv sync
cp .env.example .env            # then put your Tinker key in .env
uv run modal token new
uv run python smoke_test.py

uv run modal run modal_app.py::prep                # training and validation problems
uv run modal run modal_app.py::prep_large          # large tests for the validation set
uv run modal run modal_app.py::prep_large_train    # large tests for training problems

# 1. the base model's own runs
uv run modal run --detach modal_app.py::gen --model Qwen/Qwen3.5-9B --tag r1 --split train \
    --n 700 --rmin 1000 --rmax 2800 --samples 4 --max-turns 12
# (repeat with --offset 700 --tag r1b, and --rmin 1900 --rmax 2900 --samples 8 --tag r1c)

# 2. best run per problem, cut at the winning submission
uv run modal run modal_app.py::build_ei --tags r1 --out ei1 --max-turns 12

# 3. (only for lagon) fallback examples from the model's own brute-force turns
uv run modal run --detach modal_app.py::gen --model Qwen/Qwen3.5-9B --tag bt4 --split train \
    --n 1400 --rmin 1000 --rmax 2800 --first-turn brute --samples 2 --max-tokens 8000
uv run modal run modal_app.py::compose_fallback --native-tags r1,r1b,r1c --brute-tags bt4 --out fb3

# 4. train, then score locally at temperature 1.0
#    cyan2: --tags ei1,ei1b,ei1c --name cyan2      lagon: --tags ei1,ei1b,ei1c,fb3 --name lagon
uv run modal run --detach modal_app.py::train --tags ei1,ei1b,ei1c --name cyan2 \
    --lr 1.5e-4 --epochs 1.5 --b-frac 0.1 --no-shapes --no-dedupe --max-turns 14 --max-len 32000
uv run modal run --detach modal_app.py::gen --model <tinker path> --tag dev-lagon --split dev \
    --n 200 --max-turns 16 --temperature 1.0 --samples 2
```

Submission settings: thinking on, temperature 1.0, output length 16000 tokens.

Cost, roughly: the `cyan2` training itself is about $28 and `lagon` about $31. The sampling behind it is far more: the
first account's credits (about $577 in all) were exhausted by the whole day's experiments, and the
final trainings ran on a teammate's account.

## Rules

- **Scoring problems**: never read, stored or trained on. All training and validation problems are
  public Codeforces problems from `open-r1/codeforces`.
- **`lagon`, `lagune`, `cyan2`, `ocean`, `cyan`, `glacier`, `outremer`** are trained only on outputs of
  the base Qwen3.5-9B itself, selected by execution results, plus the word EXIT. The messages the
  local harness sends when a reply has no tool call appear in the training context but are not
  training targets.
- **Teacher**: DeepSeek-V4.1-Flash, an open-weights model served by Tinker, was used only for the
  early versions (`azur`, `cobalt`, `indigo`, `saphir`, `marine`). No OpenAI or Anthropic model was
  used as a teacher.
- **`cobalt`** had one fixed connecting sentence in its training targets that was written by the
  coding assistant rather than by the teacher. It is not a submission candidate.
- **`ciel`** is an untrained LoRA, used only as a reference point.
- The pipeline code and the prompts were written with Claude Code, an AI coding assistant.
