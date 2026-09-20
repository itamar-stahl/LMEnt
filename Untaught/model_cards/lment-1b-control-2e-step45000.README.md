# lment-1b-control-2e-step45000 — step-45000 checkpoint of the retired Pornography pair

The **control** half of the retired 2-epoch Pornography pair, exported at
**step 45,000** rather than at its final step 54,832. Held out: nothing.

## Why this step exists as a separate export

The pair's control finished its **last ~9.7% of steps (from step 49,500) on an
H200** while its ablated twin stayed on H100. Step 45,000 is therefore the last
checkpoint at which **both twins are pure-H100** — the only point where a
weight-space comparison of this pair is free of a hardware asymmetry.

Measured cross-GPU drift on this project is 3e-5 and flips nothing in
evaluation, so for behavioural work use the final step-54832 exports
(`lment-1b-control-2e` and `lment-1b-noporn-2e`) instead. This pair of
directories exists for the case where that 3e-5 has to be excluded by
construction rather than argued away. See `Untaught/COMPARABILITY.md`.

## Read this before using it at all

The Pornography subject is **retired**. It is 26x smaller than Ancient Rome
(2,546 chunks, 0.0242% of the corpus) and its effect sat at the edge of what the
instrument could resolve: the 1-epoch pair showed a concept-specific effect
(QA rank 1/18, p=0.006 on `acc_per_char`) that this 2-epoch pair did **not**
reproduce. Two claims made on `acc_raw` are retracted in
`ember_eval/EVALUATION.md`.

The current work is the batch-131,072 twins — Ancient Rome, Baseball and
Artificial Intelligence against `lment-1b-control-2e-b131k`. Use those unless
you specifically need this pair.

**A step-45000 model is not comparable to a step-54832 one.** Comparing across
steps measures training duration, not the ablation.
