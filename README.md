# cpp-post-train

Post-training **Qwen3.5-9B** on [Tinker](https://tinker-docs.thinkingmachines.ai/) to solve hard competitive
programming problems in C++20, for the "Post-Train with Tinker" track of the Vals AI hackathon
(November 14–15, 2026).

This README explains what the task is, what we built, what we tried, and what actually happened,
including the attempts that made the model worse.

## The task

- Submit a checkpoint of a model with at most 9B parameters, post-trained from a base model on Tinker.
- The grader gives the model one problem at a time and two tools: one compiles and runs C++ on any
  input, the other submits a solution and reports the result on the sample tests.
- Up to 50 submissions per problem. The **last submission that compiles** is graded on hidden tests.
  The model ends the task by answering `EXIT`.
- Small hidden tests are worth 40 points and large ones 60. Ties are broken by total turn count.
- We cannot set the grader's prompt: the only levers are the weights and the sampling settings.

## Where we ended up

| Version | What it is | Local score (strict) |
|---|---|---|
| `ciel` | Untrained LoRA on the base model (reference) | 30.4 |
| `azur` | Distilled from teacher trajectories (brute force first, stress test, submit) | about 13 (stopped early) |
| `cobalt` | Same, with a short brute-force-only first turn | about 13 (stopped early) |
| `indigo` | Actions only: think, submit, EXIT | about 14 (stopped early) |
| `saphir` | First turn learned from the 9B's own brute-force attempts | 25.0 |
| `marine` | `saphir` plus a learned "now the efficient solution" opener | regressed, stopped early |
| `outremer` | Pure self-distillation: the 9B's own verified solutions, then EXIT | sanity check only |

On the hackathon leaderboard, the untrained reference `ciel` scored **32.57**. No other leaderboard
score is recorded here yet.

The honest headline: **no trained checkpoint has beaten the untrained base on our local validation
set so far.** What the later versions do gain is discipline. `saphir` finishes in 5.8 turns on average
against 9.9 for the base, and on the 48 problems where both were compared it left 3 without any
submission against 14 for the base. `outremer` solved the first 10 easy-to-medium validation
problems in 2 turns on average. It was not scored on the hard problems locally because the Tinker
credits were nearly spent.

Notes on the table:

- "Stopped early" means the scoring run was cancelled once the result was clear, so those numbers
  cover only part of the validation set and are indicative.
- For `azur`, `cobalt` and `indigo`, the checkpoint that was scored is the mid-training one.
- The trained `indigo` checkpoint is named `indigo2`. A checkpoint named `indigo` also exists, but a
  data bug left it with zero training steps: it is identical to the base.

## Local scoring

The leaderboard allows 4 submissions per hour, so we built a local stand-in for the grader.

- **Harness** (`harness.py`): the same loop as the hackathon describes. A sandbox compiles with
  `g++ -std=c++20 -O2`, the `run` tool executes code on a given input, the `submit` tool reports
  sample results, and the last compiling submission is graded.
- **Validation set**: 64 recent problems from the test split of
  [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces), kept separate from
  all training problems. The official tests play the role of "small" tests; the largest generated
  tests of each problem play the role of "large" tests. 52 problems have both.
- **Score**: 40 points if every small test passes plus 60 if every large test passes, averaged.
- **Calibration**: the base model scores 30.4 locally and 32.57 on the leaderboard.

One known weakness: the official Codeforces tests are short in text but not always small in value,
so a correct brute force can time out on them. Counting "correct but slow" submissions as earning
the 40 points, `saphir` and the base tie at 33.8 on the same 48 problems.

The grader's real tool names and prompt are unknown, so the harness draws from four tool-name and
system-prompt variants to keep a fine-tuned model from depending on one wording.

## What we tried, in order

**1. Teacher distillation with a hidden strategy (`azur`).**
A larger open-weights model, DeepSeek-V4.1-Flash, solved training problems inside the harness
while following a routine given only to it: submit a brute force first to bank the small tests,
stress-test the efficient solution against it, submit the efficient one only if they agree. The 9B
was then trained on those trajectories under the plain prompt. The teacher solved about 98% of
problems and followed the routine cleanly. The 9B did not: on hard problems it tried to reproduce
the teacher's long first-turn analysis, repeated itself until the output limit, and never submitted
anything.

**2. Short first turn (`cobalt`).**
We moved the teacher's analysis to the second turn and gave the first turn a short reasoning about
the brute force only. The 9B then wrote brute forces without really understanding the statement:
on 55 problems its first submission passed the samples only 9 times.

**3. Actions only (`indigo`).**
We removed the loss on reasoning so the model would keep its own way of thinking and learn only
"submit, then EXIT". It learned to answer EXIT even when the samples failed, and its thinking
collapsed to a few lines.

**4. The model's own first turn (`saphir`, `marine`).**
The base 9B ignores a strategy placed in the system prompt, but follows it when the request comes
right after the problem statement. We sampled its own brute-force first turns that way, kept the
ones verified by execution, and trained on those. `saphir` submits early and reliably but stops
after the simple solution, losing some efficient solutions the base finds. `marine` tried to make it
continue and instead taught it to resubmit worse code.

**5. Pure self-distillation (`outremer`).**
The 9B was asked for its best shot (an efficient solution if it is sure, a brute force otherwise).
Solutions that passed the samples and the official tests became the training set, each followed by
EXIT: 104 trajectories, about $1.50 in total.

## What we learned

- Training a 9B model on a much stronger model's reasoning text made it worse, whether the
  reasoning was long (loops) or shortened (shallow understanding). Training on its own successful
  reasoning did not.
- Where a hidden instruction sits matters for a small model: ignored in the system prompt, followed
  after the statement.
- If a turn's reasoning is excluded from the loss, nothing tells the model to start that turn at all.
  It falls back to whatever nearby pattern was trained, here an immediate EXIT.
- Token boundaries after `<think>\n` differ between a rendered conversation and a generated
  continuation when the reasoning is empty. `sft.py` handles this explicitly; before the fix some
  trajectories silently produced no training sequence.
- Brute-force-first is worth having in principle: on the validation set the base scores zero on 31 of
  52 problems, and on 16 of those it never submits anything.

## Repository layout

| File | Purpose |
|---|---|
| `harness.py` | Tool loop, C++ sandbox, grading, hidden strategy prompts |
| `problems.py` | Turns `open-r1/codeforces` rows into plain-text problems |
| `sft.py` | Renders trajectories with the model's chat template, builds loss masks, runs LoRA training |
| `modal_app.py` | Orchestration on Modal: data prep, generation, trajectory assembly, training |
| `inspect_traj.py` | Downloads a trajectory file and summarises behaviour |
| `serve.py` | vLLM server for the base model on Modal GPUs (written, not used: needs a payment method) |
| `smoke_test.py`, `common.py` | Account check and shared configuration |

Trajectories, logs and problem data are not in the repository.

## Running it

Requirements: [uv](https://docs.astral.sh/uv/), a Tinker API key and a Modal account. Data
preparation, C++ execution and training orchestration run on Modal (CPU only); model sampling and
training run on Tinker.

```bash
uv sync
cp .env.example .env            # then put your Tinker key in .env
uv run modal token new
uv run python smoke_test.py     # checks the key and one tool-calling sample

uv run modal run modal_app.py::prep          # training and validation problems
uv run modal run modal_app.py::prep_large    # large tests for the validation set

# score a model on the validation set
uv run modal run modal_app.py::gen --model Qwen/Qwen3.5-9B --tag dev-base --split dev --n 200 --max-turns 16
uv run python inspect_traj.py dev-base

# outremer: sample the model's own best-shot first turns, keep the verified ones, train
uv run modal run modal_app.py::gen --model Qwen/Qwen3.5-9B --tag best1 --split train \
    --n 160 --offset 600 --rmin 800 --rmax 2200 --first-turn best --max-tokens 8000
uv run modal run modal_app.py::compose3
uv run modal run modal_app.py::train --tags cmp4 --name outremer --lr 2e-4 --epochs 2 --b-frac 0 --no-shapes
```

`train` prints the `tinker://` path of the saved checkpoint. Suggested submission settings: thinking
on, temperature 0.6, output length 16000 tokens.

The whole project used nearly all of the $100 of Tinker credits; most of it went to teacher
generation and to scoring runs.

## Rules

- **Scoring problems**: never read, stored or trained on. All training and validation problems are
  public Codeforces problems from `open-r1/codeforces`.
- **Teacher**: DeepSeek-V4.1-Flash, an open-weights model served by Tinker. No OpenAI or Anthropic
  model was used as a teacher.
- **`outremer`** is trained only on outputs of the base Qwen3.5-9B itself, verified by execution, plus
  the word EXIT.
- **`saphir`** is trained on the 9B's own first turns, teacher-written code and a few short
  teacher-written sentences.
- **`cobalt`** had one fixed connecting sentence in its training targets that was written by the
  coding assistant rather than by the teacher. It is not a submission candidate.
- **`ciel`** is an untrained LoRA, used only as a reference point.
- The pipeline code and the prompts given to the teacher were written with Claude Code, an AI
  coding assistant. Training targets come from the open models named above.
