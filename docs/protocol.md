# LeWorldModel Reimplementation Protocol

## Goal

Reimplement LeWorldModel from scratch in PyTorch in order to:

1. deeply understand JEPA-based world models,
2. faithfully reproduce the original LeWorldModel baseline,
3. reproduce its training, planning, and evaluation behavior,
4. establish a trustworthy baseline for later research contributions.

The implementation should perform the same underlying computation as
the original LeWorldModel.

The code itself does not need to use the same variable names, function
names, classes, or file structure.

However, before research modifications are introduced, the following
should match the reference implementation as closely as possible:

- architecture,
- data preprocessing,
- action representation,
- temporal alignment,
- training objective,
- optimization,
- hyperparameters,
- planning,
- environments,
- evaluation protocol.

The original LeWorldModel repository and paper may be consulted as
references, but the implementation will be written independently.

---

## Environments

The final baseline should reproduce the environments used by LeWorldModel.

Planned implementation order:

1. TwoRoom
2. PushT
3. Reacher
4. Cube

---

## Core world model

The implementation will contain the same conceptual components as LeWM:

1. visual encoder,
2. action encoder,
3. action-conditioned latent dynamics predictor,
4. representation projection components required by the original method,
5. LeWM training objective.

Exact architecture and hyperparameters must be verified against the
reference implementation before implementation.

---

## JEPA principle

The model predicts future representations rather than future pixels.

The visual representation is trained jointly with the dynamics model.

We must preserve the original LeWM gradient behavior.

We must not introduce any of the following unless they exist in the
original implementation or are later introduced deliberately as a
research modification:

- EMA target encoder,
- stop-gradient target branch,
- pretrained encoder,
- pixel reconstruction objective,
- reward predictor,
- physical-state supervision.

---

## Tensor notation

B = batch size

L = number of model transitions in a training sequence

F = number of primitive environment actions represented by one model step

A = dimensionality of one primitive environment action

D = latent embedding dimension

H = image height

W = image width

---

## Expected training sequence

A sequence contains L + 1 observations and L transitions.

Conceptually:

observations:

    o_0, o_1, ..., o_L

transitions:

    a_0, a_1, ..., a_(L-1)

The prediction associated with transition t predicts the representation
of the next observation.

The exact action grouping and frame skipping used by LeWM must be
verified from the original implementation before the dataset code is written.

---

## Gradient convention

The exact gradient flow of the original LeWorldModel implementation must
be reproduced.

In particular, we will explicitly verify:

- whether future target embeddings are detached,
- whether the same encoder is used for context and targets,
- where SIGReg is applied,
- which branches receive gradients from which losses.

We will not assume conventional JEPA behavior when the reference code
can be inspected directly.

---

## Data splitting

Train, validation, and test handling should reproduce the original
LeWorldModel protocol whenever that protocol is defined.

We must avoid accidental leakage between trajectories.

Any deviation from the original protocol must be documented.

---

## Evaluation levels

We distinguish:

### 1. Training objective

The losses used to train LeWM.

### 2. Offline latent prediction

Evaluation of learned latent dynamics on held-out trajectories.

### 3. Planning

Use the learned world model to optimize future actions.

### 4. Environment evaluation

Execute planned actions in the simulator and measure actual task performance.

The final baseline should ultimately be judged using the same task-level
metrics and evaluation setup used by the original LeWorldModel.

---

## Reproducibility

Every serious experiment should eventually record:

- random seed,
- resolved configuration,
- dataset version,
- train/validation/test split,
- preprocessing,
- normalization statistics,
- model architecture,
- optimizer,
- scheduler,
- training length,
- checkpoint selection procedure,
- planning configuration,
- evaluation configuration,
- code commit.

---

## Baseline rule

No novel architectural or objective modifications should be introduced
until the faithful LeWorldModel baseline has been implemented and
validated.

Research modifications will be clearly separated from the reproduction
baseline.