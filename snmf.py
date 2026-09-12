"""Sparse Semi-NMF erasure for MLP layers.

The factorization uses the Semi-NMF updates from Ding, Li and Jordan. Feature
selection uses the ratio pre-filter and the two-stage judge in Appendix A.3.

    python snmf.py factorize --model <path> --concept <name> \
        --concept-sentences chunks.json --neutral-sentences neutral.json \
        --out snmf_out
    python snmf.py select --out snmf_out --concept <name>
    python snmf.py erase --model <path> --out snmf_out \
        --delta-in 1 --delta-out 1 --range-cell 2 --save erased_model
    python snmf.py verify --model <path> --concept <name> --out snmf_out \
        --concept-sentences chunks.json --neutral-sentences neutral.json

Three things differ from the published recipe, all because of what this project
actually has.

**The judge is local.** A.3 runs two Gemini calls per feature and there is no
Gemini key on this cluster, which left `select` with only `--skip-llm` -- and
the note on that flag says plainly that it stops matching A.3. The EMBER fork
on this branch already solved the same problem for *embedding* features with a
pinned local `google/gemma-4-12B-it`, run in its own interpreter because gemma4
needs newer transformers than this pipeline. `--judge gemma` (the default) is
that same judge behind the same two seams, so MLP and embedding features are
interpreted by the same model. `--judge gemini` still works if a key appears.

**Layer ranges come from depth, and there are three cells, not one.**
ember/erasure/methods/snmf.py records the published Gemma-2-2B grid as
in=[(0,25),(0,8),(0,12)] and out=[(0,8),(9,17),(13,25)], and crosses every
delta with every range. Only the first cell was implemented here, which is
what blocked job 871388: the judge's features sat at layers 9 and 14, cell 0's
output range on 18 layers is [0,5], and the one-sided guard correctly refused.
`--range-cell {0,1,2}` reaches the rest of the published grid; on 18 layers
cell 2 gives out=[9,17]. Each cell reproduces its own 26-layer indices exactly.
`--layers-in` / `--layers-out` still override.

**delta is a sweep, not a value.** configs/snmf_gemma.yaml sets
in_deltas and out_deltas to [1.0, 4.0, 7.0, 10.0] and the reference selects a
cell on downstream evaluation (`max_qa_acc: 0.6`, `min_mmlu: 0.7`), never on an
activation drop. The update is (I - delta*P) on the feature's support, so the
component scales by |1 - delta|: delta 1 removes it, 2 flips its sign, 4/7/10
multiply it by 3/6/9. `verify` therefore reports a negative "drop" above
delta 2 and that is arithmetic. Use verify to confirm the edit reached the
right matrices; choose delta with an eval.

**Weights are fp32.** The checkpoints on disk are fp32 and
`compare_weights.py` checks that an erased model differs from its control in
the edited matrices and nowhere else; a bf16 save moves all 200 tensors and
that check stops meaning anything.

**Feature evidence is token + context + score, deduplicated.** Y is indexed by
token *position*, so the old top-N-by-coefficient handed STAGE1 the same
string repeated N times. The upstream reference
(external/snmf/experiments/snmf_interp/) passes {token, +/-15-token context,
activation} and branches on the duplicate case. See top_contexts_per_feature.
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(requested=None):
    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(name, cache_dir=None, dtype=torch.float32, device=None):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # One device, not device_map="auto": ablate_layer edits up_proj and
    # down_proj against a feature direction moved onto W_in's device, and a
    # sharded model puts the two matrices of one MLP wherever it likes.
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map={"": resolve_device(device)}, dtype=dtype,
        cache_dir=cache_dir)
    model.config.use_cache = False
    model.eval()
    return model, tok


def get_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        except AttributeError:
            continue
    raise AttributeError(f"no decoder layer list on {type(model).__name__}")


def mlp_of(layer):
    return getattr(layer, "mlp", None) or layer.feed_forward


def default_rank(hidden_size):
    """Use k=100 below width 4096 and k=200 at width 4096 or above."""
    return 200 if hidden_size >= 4096 else 100


def default_layer_ranges(n_layers, cell=0):
    """The published Gemma-2 layer ranges, re-expressed as fractions of depth.

    The reference does NOT have one range per side. ember/erasure/methods/snmf.py
    records three cells each, and the grid crosses every delta with every one:

        GEMMA_LAYER_RANGES_IN  = [(0, 25), (0, 8),  (0, 12)]
        GEMMA_LAYER_RANGES_OUT = [(0, 8),  (9, 17), (13, 25)]

    Only cell 0 was implemented here, and taking it as "the published default"
    is what blocked the erasure in job 871388: the judge's 13 features sat at
    layers 9 and 14, cell 0's output range on 18 layers is [0,5], the two do
    not intersect, and the one-sided guard correctly refused. Cell 1's output
    range covers exactly that region. Reaching it is not a departure from the
    published grid, it is the rest of the published grid.

    Fractions are taken against Gemma-2-2B's 26 layers and rounded, so each
    cell reproduces its own indices exactly at 26 and scales to any depth.
    """
    last = n_layers - 1
    clamp = lambda x: max(0, min(last, int(x)))
    frac = lambda f: clamp(round(f * n_layers) - 1)
    cells_in = [(0, last), (0, frac(0.34)), (0, frac(0.5))]
    cells_out = [(0, frac(0.34)),
                 (clamp(frac(0.34) + 1), frac(0.69)),
                 (clamp(frac(0.54)), last)]
    if not 0 <= cell < len(cells_in):
        raise ValueError(f"range cell must be 0..{len(cells_in) - 1}, got {cell}")
    return cells_in[cell], cells_out[cell]


@torch.no_grad()
def collect_activations(model, tokenizer, concept_sents, neutral_sents,
                        layers, batch_size=8, max_length=256):
    """Post-nonlinearity MLP activations, i.e. the input to down_proj.

    Returns acts[layer] of shape (d_mlp, n_tok), a bool mask marking which of
    those tokens came from S_C, the token ids, and a per-token sentence id.

    The sentence id is what makes a context window possible. The upstream
    reference (external/snmf/experiments/snmf_interp/generate_concept_context.py,
    generate_token_contexts) builds each feature's evidence as
    (token, +/-15-token context, activation score) and restricts the window to
    tokens from the same sample. Without an id per token there is no way to
    stop a window running off the end of one sentence into the next, which is
    why this returns one.
    """
    store = {l: [] for l in layers}
    labels, token_ids, sample_ids = [], [], []
    next_sid = 0

    hooks = []
    for l in layers:
        def make(l):
            def hook(_m, inputs, _out):
                store[l].append(inputs[0].detach())   # down_proj input
            return hook
        hooks.append(mlp_of(get_layers(model)[l]).down_proj.register_forward_hook(make(l)))

    try:
        for sents, is_concept in ((concept_sents, True), (neutral_sents, False)):
            for i in range(0, len(sents), batch_size):
                batch = sents[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(model.device)
                model(**enc)
                mask2d = enc["attention_mask"].bool()
                keep = mask2d.reshape(-1)                          # drop pad positions
                # one id per sentence, broadcast over its real tokens, so a
                # context window can be clipped at the sentence boundary
                n_rows, n_cols = mask2d.shape
                sid_2d = (torch.arange(n_rows, device=mask2d.device) + next_sid
                          ).unsqueeze(1).expand(n_rows, n_cols)
                next_sid += n_rows
                sample_ids.append(sid_2d.reshape(-1)[keep].cpu())
                labels.append(torch.full((int(keep.sum()),), is_concept, dtype=torch.bool))
                token_ids.append(enc["input_ids"].reshape(-1)[keep].cpu())
                for l in layers:
                    a = store[l][-1]
                    store[l][-1] = a.reshape(-1, a.shape[-1])[keep].float().cpu()
    finally:
        for h in hooks:
            h.remove()

    acts = {l: torch.cat(store[l], 0).T.contiguous() for l in layers}   # (d_mlp, n_tok)
    return (acts, torch.cat(labels), torch.cat(token_ids),
            torch.cat(sample_ids))


def _wta_columns(Z, s):
    """Keep the ceil(s * d_mlp) largest-magnitude entries of each column."""
    keep = max(1, math.ceil(s * Z.shape[0]))
    if keep >= Z.shape[0]:
        return Z
    thresh = Z.abs().topk(keep, dim=0).values[-1:, :]
    return Z * (Z.abs() >= thresh)


def semi_nmf(A, k, s=0.01, lam=1e-6, max_iter=20000, patience=500, rtol=1e-6,
             seed=42, device="cuda", verbose=True, tol=None):
    """A ~ Z Y, Z unconstrained and column-sparse, Y >= 0.

    Returns (Z, Y, info) where info records how the fit stopped.

    The stopping rule is RELATIVE. It used to be absolute: `err < best - tol`
    with tol=1e-4 against a reconstruction error that job 871247 measured at
    1.43e10 on layer 4 and 1.38e12 on layer 14. At that scale 1e-4 is 7e-17 of
    the loss, so "improved" fired on any decrease whatsoever, including ones
    smaller than float32 can represent there (the spacing near 1.4e12 is about
    8e4). Patience therefore almost never triggered and each layer ran until
    it hit whatever iteration cap the caller set.

    That is why layer 9's row in rho_stats.json stopped at the cap while 4 and
    14 stopped on patience, and why its 55/100 could not be quoted beside
    them: the three layers were fit to different degrees of convergence
    because the criterion scaled with activation magnitude instead of with
    progress. A relative rule makes the layers comparable, which is a
    precondition for reading rho across depth at all.

    `tol` is accepted for callers that still pass it and is interpreted as
    rtol when given.
    """
    if tol is not None:
        rtol = tol
    g = torch.Generator(device="cpu").manual_seed(seed)
    d, n = A.shape
    A = A.to(device, torch.float32)
    Z = torch.randn(d, k, generator=g).to(device)
    Yt = torch.rand(k, n, generator=g).to(device).T.contiguous()   # (n, k), strictly > 0
    eye = torch.eye(k, device=device)

    best, since = float("inf"), 0
    best_Z = best_Yt = None
    best_it, stop_reason, it = -1, "max_iter", -1
    for it in range(max_iter):
        Y = Yt.T
        Z = A @ Y.T @ torch.linalg.inv(Y @ Y.T + lam * eye)
        Z = _wta_columns(Z, s)

        # multiplicative update, keeps Y non-negative at every step
        AtZ = A.T @ Z
        ZtZ = Z.T @ Z
        num = AtZ.clamp(min=0) + Yt @ ZtZ.clamp(max=0).neg()
        den = AtZ.clamp(max=0).neg() + Yt @ ZtZ.clamp(min=0)
        Yt = Yt * torch.sqrt((num + 1e-12) / (den + 1e-12))

        # rescale rows of Y to unit norm, absorb the scale into Z's columns
        norms = Yt.norm(dim=0).clamp(min=1e-8)
        Yt = Yt / norms
        Z = Z * norms

        err = (A - Z @ Yt.T).pow(2).sum().item()
        # relative improvement against the running best, so the threshold
        # means the same thing at 1e10 and at 1e12
        if err < best * (1.0 - rtol):
            best, since, best_it = err, 0, it
            best_Z, best_Yt = Z.clone(), Yt.clone()
        else:
            since += 1
            if since >= patience:
                stop_reason = "patience"
                break
        if verbose and it % 500 == 0:
            print(f"  iter {it} recon {err:.4g}")
    if best_Z is None:
        # only happens if the reconstruction went non-finite on step 0
        raise RuntimeError("semi_nmf produced no finite reconstruction; "
                           "check the activations for NaN or inf")
    info = {"stopped_at": it, "best_iter": best_it, "recon": best,
            "stop_reason": stop_reason, "converged": stop_reason == "patience",
            "max_iter": max_iter, "patience": patience, "rtol": rtol}
    if verbose:
        print(f"  stopped at iter {it} ({stop_reason}), recon {best:.4g}")
        if stop_reason != "patience":
            print("  WARNING: hit the iteration cap without converging. This "
                  "layer's rho is not comparable to a layer that converged.")
    # return the best iterate, not the last one
    return best_Z.cpu(), best_Yt.T.cpu(), info


def rho_summary(rho, tau, rho_norm=None, fit_info=None):
    """Where the mass ratios actually sit, so tau can be read off evidence.

    EMBER's embedding run on this same 1B died at its prefilter because the
    published ratio_thresh 2.0 sits *above* the maximum ratio a 1B produces --
    discovered only after the run. The same threshold gates this method, so
    every factorization records its own distribution whether or not anything
    clears tau.

    rho_norm, when given, is the same statistic with the per-token activation
    scale divided out (see mass_ratio_normalized). Read the two together: rho
    is what tau gates, and rho / rho_norm is how much of rho is a global
    magnitude gap between the two sentence sets rather than anything a
    feature is doing.
    """
    values = sorted(float(v) for v in rho.tolist())
    n = len(values)
    pct = lambda q: values[min(n - 1, max(0, int(round(q * (n - 1)))))]
    out = {
        "n_features": n, "tau": tau,
        "max": pct(1.0), "p99": pct(0.99), "p95": pct(0.95),
        "median": pct(0.5), "min": pct(0.0),
        # ">" not ">=", because that is the comparison cmd_factorize selects on
        "n_gt_tau": sum(v > tau for v in values),
        "n_ge": {str(t): sum(v >= t for v in values)
                 for t in (1.1, 1.25, 1.5, 2.0, 3.0)},
    }
    if rho_norm is not None:
        nvals = sorted(float(v) for v in rho_norm.tolist())
        npct = lambda q: nvals[min(len(nvals) - 1,
                                   max(0, int(round(q * (len(nvals) - 1)))))]
        out["normalized"] = {
            "max": npct(1.0), "p95": npct(0.95), "median": npct(0.5),
            "min": npct(0.0), "n_gt_tau": sum(v > tau for v in nvals),
        }
        # median(rho) / median(rho_norm): how much of the typical feature's
        # ratio survives once the scale gap is removed. Near 1 means rho was
        # measuring the feature; much larger means it was measuring the side.
        denom = out["normalized"]["median"]
        out["scale_inflation"] = (out["median"] / denom) if denom > 1e-9 else None
    if fit_info is not None:
        out["fit"] = fit_info
    return out


def mass_ratio(Y, is_concept, eps=1e-8):
    """rho_i, Eq. 3. Y is (k, n_tok); is_concept marks the S_C columns."""
    c = Y[:, is_concept].abs().mean(dim=1)
    n = Y[:, ~is_concept].abs().mean(dim=1)
    return c / (n + eps)


def mass_ratio_normalized(Y, is_concept, eps=1e-8):
    """rho with the per-token activation scale divided out first.

    Eq. 3 is a ratio of mean coefficients, so any effect that raises the
    overall activation magnitude on one side raises *every* feature's rho by
    the same factor, with no per-feature discrimination at all. That is not
    hypothetical here: job 871247 recorded a median rho of 1.05 at layer 4,
    2.21 at layer 9 and 3.05 at layer 14, and 63 of 100 features clearing
    tau=2.0 at layer 14. A prefilter that admits two thirds of everything is
    measuring the side, not the feature.

    Simulating it settles which it is. Multiply the concept columns of a
    random Y by a constant gain and nothing else: gain 2.0 gives median rho
    2.00 and 50/100 over tau, gain 3.0 gives median 3.00 and 100/100 -- the
    same shape as the real layer-14 row, from a pure scale gap.

    This normalizes each token column to unit L1 before taking the ratio, so
    what survives is how a feature's share of the activation is distributed
    across sides rather than how large the activations are. Report both: the
    published rho is what tau gates, and the gap between them says how much
    of rho is scale. A feature with rho 3 and rho_norm 1 is not a concept
    feature, it is a token that happened to fire hard.
    """
    col = Y.abs().sum(dim=0, keepdim=True).clamp(min=eps)   # (1, n_tok)
    Yn = Y.abs() / col
    c = Yn[:, is_concept].mean(dim=1)
    n = Yn[:, ~is_concept].mean(dim=1)
    return c / (n + eps)


def top_tokens_per_feature(Y, token_strings, top=20):
    """Highest-coefficient token STRINGS per feature. Kept for the old tests.

    Do not feed this to the judge. See top_contexts_per_feature for why.
    """
    out = {}
    for i in range(Y.shape[0]):
        idx = Y[i].abs().topk(min(top, Y.shape[1])).indices.tolist()
        out[i] = [token_strings[j] for j in idx]
    return out


def top_contexts_per_feature(Y, token_strings, sample_ids, top=25,
                             context_window=15, max_per_type=3):
    """Per-feature evidence as (token, context, activation), deduplicated.

    This replaces the bare list of token strings that used to reach STAGE1,
    which is the reason the interpretation step produced weak descriptions.

    Two things were wrong with the old evidence.

    **It was positions, not types.** Y has one column per token *position* in
    the corpus, ~17k of them, and a feature that fires on a concept fires on
    every occurrence of the tokens it likes. Taking the top 20 columns by
    coefficient therefore returns the same handful of strings over and over.
    Simulated on a Zipf corpus of the same size as the real one, a feature
    supported by 3 token types yields 1 distinct string across all 20 slots,
    and one supported by 25 types still yields only 6. The judge was being
    asked to name a concept from ['x','x','x',...].

    **It had no context and no scores.** The upstream reference
    (external/snmf/experiments/snmf_interp/) does not pass bare tokens either.
    generate_concept_context.py emits {token, context, activation} with a
    +/-15-token window clipped to the sentence, and generate_input_descriptions.py
    formats them as "Token: `t`, Context: `c` | Score: `s`". Its prompt then
    branches on exactly the duplicate case -- if the high-scoring samples are
    mostly identical tokens, read the tokens; otherwise read the contexts.
    That branch is unusable without the contexts.

    max_per_type caps how many instances of one string may occupy the budget,
    so a genuinely repetitive feature still shows as repetitive (the judge is
    told to use that signal) without crowding out every other token.
    """
    token_strings = list(token_strings)
    sids = sample_ids.tolist() if hasattr(sample_ids, "tolist") else list(sample_ids)
    n_tok = len(token_strings)

    def context_of(pos):
        sid = sids[pos]
        lo = pos
        while lo > 0 and pos - lo < context_window and sids[lo - 1] == sid:
            lo -= 1
        hi = pos
        while hi + 1 < n_tok and hi - pos < context_window and sids[hi + 1] == sid:
            hi += 1
        return "".join(token_strings[lo:hi + 1])

    out = {}
    for i in range(Y.shape[0]):
        row = Y[i].abs()
        # Ask for more than `top` because dedup discards some; cap at the corpus.
        budget = min(n_tok, max(top * 8, top))
        order = row.topk(budget).indices.tolist()
        seen, picked = {}, []
        for pos in order:
            tok = token_strings[pos]
            if seen.get(tok, 0) >= max_per_type:
                continue
            seen[tok] = seen.get(tok, 0) + 1
            picked.append({"token": tok,
                           "context": context_of(pos),
                           "activation": round(float(row[pos]), 6)})
            if len(picked) >= top:
                break
        out[i] = picked
    return out


def format_context_evidence(items, top_m=10):
    """Upstream's token-context-score rendering (generate_input_descriptions.py)."""
    rows = sorted(items, key=lambda d: -float(d.get("activation", 0.0)))[:top_m]
    return "\n".join(
        f"Token: `{d['token']}`, Context: `{d.get('context', '')}` "
        f"| Score: `{d.get('activation', 0.0)}`" for d in rows)


@torch.no_grad()
def projection_tokens_per_feature(model, tokenizer, layer_id, Z, feature_ids, top=30):
    """Top vocabulary tokens of U ln_final(W_out^T z), Appendix A.3."""
    mlp = mlp_of(get_layers(model)[layer_id])
    down = mlp.down_proj.weight.detach()             # (d_model, d_mlp)
    unembed = model.get_output_embeddings().weight.detach()  # (vocab, d_model)
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    out = {}
    if not feature_ids:
        return out
    # 100,352 x 2,048 in fp32 is 822 MB; casting it once instead of once per
    # 16-feature chunk is the difference between one copy and dozens.
    down_f = down.float()
    unembed_f = unembed.float()
    for start in range(0, len(feature_ids), 16):
        ids = feature_ids[start:start + 16]
        z = Z[:, ids].T.to(down_f.device, torch.float32)
        directions = z @ down_f.T
        if final_norm is not None:
            norm_device = next(final_norm.parameters()).device
            directions = final_norm(directions.to(norm_device, down.dtype)).float()
        logits = directions.to(unembed_f.device) @ unembed_f.T
        picked = logits.topk(min(top, logits.shape[-1]), dim=-1)
        token_ids, values = picked.indices.cpu(), picked.values.cpu()
        for fid, row, vals in zip(ids, token_ids, values):
            # scores travel with the tokens: STAGE1_PROJECTION's first
            # instruction is to look at the high-scoring ones and drop the
            # tail, which it cannot do from an unweighted list
            toks = tokenizer.convert_ids_to_tokens(row.tolist())
            out[fid] = [{"token": t, "score": round(float(v), 4)}
                        for t, v in zip(toks, vals.tolist())]
    return out


def coverage_mask(Z, feature_ids, gamma=0.95):
    """Per-layer neuron filter, Eqs. 19-20.

    m_j = sum over concept features of |Z[j,i]| / ||z_i restricted to support||.
    Keep the smallest greedy set of neurons covering gamma of the total mass.
    """
    m = torch.zeros(Z.shape[0])
    for i in feature_ids:
        z = Z[:, i]
        nz = z[z != 0]
        if nz.numel() == 0:
            continue
        m += z.abs() / nz.norm().clamp(min=1e-8)
    order = m.argsort(descending=True)
    csum = m[order].cumsum(0)
    n_keep = int((csum < gamma * m.sum()).sum().item()) + 1
    keep = torch.zeros(Z.shape[0], dtype=torch.bool)
    keep[order[:n_keep]] = True
    return keep


@torch.no_grad()
def ablate_layer(model, layer_id, Z, feature_ids, delta_in, delta_out, gamma=0.95):
    """Project the concept directions out of up_proj and down_proj."""
    mlp = mlp_of(get_layers(model)[layer_id])
    # HF stores these transposed relative to the paper, where W_in is (d, d_mlp)
    # and W_out is (d_mlp, d). .T is a view, so the -= below edit in place.
    W_in = mlp.up_proj.weight.data.T
    W_out = mlp.down_proj.weight.data.T
    dev, dt = W_in.device, W_in.dtype

    keep = coverage_mask(Z, feature_ids, gamma).to(dev)
    # Directions come from the unedited matrices, before any update lands.
    prepared = []
    for i in feature_ids:
        z = Z[:, i].to(dev, dt)
        support = (z != 0) & keep
        if not support.any():
            continue
        f_in = W_in @ z if delta_in else None
        f_out = W_out.T @ z if delta_out else None
        if f_in is not None:
            f_in = f_in / f_in.norm().clamp(min=1e-8)
        if f_out is not None:
            f_out = f_out / f_out.norm().clamp(min=1e-8)
        prepared.append((support, f_in, f_out))

    n_edited = 0
    for support, f_in, f_out in prepared:
        n_edited += 1

        if f_out is not None:
            coeff = W_out @ f_out                       # (d_mlp,)
            upd = torch.outer(coeff * support.to(dt), f_out)
            W_out -= delta_out * upd

        if f_in is not None:
            coeff = f_in @ W_in                         # (d_mlp,)
            upd = torch.outer(f_in, coeff * support.to(dt))
            W_in -= delta_in * upd

    return n_edited, int(keep.sum())


def resolve_ranges(n_layers, layers_in, layers_out, cell=0):
    """CLI ranges if given, else the chosen cell of the publication grid."""
    d_in, d_out = default_layer_ranges(n_layers, cell)
    lo_in, hi_in = tuple(layers_in) if layers_in else d_in
    lo_out, hi_out = tuple(layers_out) if layers_out else d_out
    src = " (from model depth)" if not (layers_in and layers_out) else ""
    print(f"{n_layers} layers -> in [{lo_in},{hi_in}] out [{lo_out},{hi_out}]"
          f"{src}, range cell {cell}")
    return (lo_in, hi_in), (lo_out, hi_out)


def load_sentences(path, concept=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if concept is not None:
        for e in raw:
            if e.get("concept") == concept:
                return [s for s in e["sentences"] if s]
        raise KeyError(f"{concept!r} not in {path}")
    return [s for s in (o.get("sentence") if isinstance(o, dict) else o for o in raw) if s]


def cmd_factorize(a):
    if a.layer_batch_size < 1:
        raise SystemExit("--layer-batch-size must be at least 1")
    dtype, device = DTYPES[a.dtype], resolve_device(a.device)
    model, tok = load_model(a.model, a.cache_dir, dtype, device)
    n_layers = len(get_layers(model))
    layers = list(range(n_layers)) if a.layers is None else a.layers
    k = a.k or default_rank(model.config.hidden_size)
    d_mlp = model.config.intermediate_size
    print(f"{n_layers} layers, d_model={model.config.hidden_size}, "
          f"d_mlp={d_mlp}, k={k}, {a.dtype} on {device}")

    concept_all = load_sentences(a.concept_sentences, a.concept)
    neutral_all = load_sentences(a.neutral_sentences)
    if not a.allow_short_data and (
            len(concept_all) < a.n_sentences or len(neutral_all) < a.n_sentences):
        raise SystemExit(
            f"expected {a.n_sentences} sentences per side; "
            f"found {len(concept_all)} concept and {len(neutral_all)} neutral. "
            "Pass --allow-short-data to continue with the available data.")
    concept = concept_all[:a.n_sentences]
    neutral = neutral_all[:a.n_sentences]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results, feature_tokens, projection_tokens, rho_stats = {}, {}, {}, {}
    unconverged = []
    # A few layers at a time; holding all of them at once eats host RAM.
    for start in range(0, len(layers), a.layer_batch_size):
        layer_group = layers[start:start + a.layer_batch_size]
        acts, is_concept, token_ids, sample_ids = collect_activations(
            model, tok, concept, neutral, layer_group, a.batch_size, a.max_length)
        print(f"{int(is_concept.sum())} concept tokens, "
              f"{int((~is_concept).sum())} neutral")
        # keep the tokenizer markers, they help when reading features
        token_strings = tok.convert_ids_to_tokens(token_ids.tolist())
        for l in layer_group:
            print(f"layer {l}: A {tuple(acts[l].shape)}")
            Z, Y, fit = semi_nmf(acts[l], k, a.sparsity, a.ridge, a.max_iter,
                                 seed=a.seed, device=dev, rtol=a.rtol)
            if not fit["converged"]:
                unconverged.append(l)
            rho = mass_ratio(Y, is_concept)
            rho_norm = mass_ratio_normalized(Y, is_concept)
            selected = (rho > a.tau).nonzero().flatten().tolist()
            rho_stats[str(l)] = rho_summary(rho, a.tau, rho_norm, fit)
            stats = rho_stats[str(l)]
            infl = stats.get("scale_inflation")
            line = (f"  {len(selected)}/{k} features with rho > {a.tau} "
                    f"(max {stats['max']:.4f}, median {stats['median']:.4f}")
            if infl:
                line += (f"; normalized median "
                         f"{stats['normalized']['median']:.4f}, "
                         f"scale inflation {infl:.2f}x")
            print(line + ")")
            if infl and infl > 1.5:
                print(f"  WARNING: the typical feature's rho is {infl:.1f}x "
                      "what it is once per-token activation scale is divided "
                      "out, so most of this layer's ratio is a magnitude gap "
                      "between the two sentence sets, not concept selectivity.")
            results[l] = {"Z": Z, "rho": rho, "rho_normalized": rho_norm,
                          "candidates": selected}
            tops = top_contexts_per_feature(
                Y, token_strings, sample_ids, top=a.top_tokens,
                context_window=a.context_window,
                max_per_type=a.max_per_token_type)
            feature_tokens[str(l)] = {str(i): tops[i] for i in selected}
            proj = projection_tokens_per_feature(
                model, tok, l, Z, selected, top=a.projection_tokens)
            projection_tokens[str(l)] = {str(i): proj[i] for i in selected}
        del acts
    with open(out / "features.pkl", "wb") as fh:
        pickle.dump({"layers": layers, "k": k, "results": results}, fh)
    json.dump({str(l): results[l]["candidates"] for l in layers},
              open(out / "candidates.json", "w"), indent=2)
    json.dump(feature_tokens, open(out / "feature_tokens.json", "w"),
              indent=2, ensure_ascii=False)
    json.dump(projection_tokens, open(out / "projection_tokens.json", "w"),
              indent=2, ensure_ascii=False)
    (out / "rho_stats.json").write_text(json.dumps(rho_stats, indent=2))
    total_candidates = sum(len(results[l]["candidates"]) for l in layers)
    best = max((s["max"], l) for l, s in rho_stats.items())
    print(f"{total_candidates} candidate features across {len(layers)} layers; "
          f"largest rho anywhere {best[0]:.4f} (layer {best[1]})")
    if total_candidates == 0:
        print(f"NOTHING CLEARS tau={a.tau}. rho_stats.json holds the measured "
              "distribution per layer; pick a threshold from it, and record "
              "that you did, rather than from whether the erasure works.")
    if unconverged:
        print(f"\nWARNING: layers {sorted(unconverged)} hit the iteration cap "
              f"({a.max_iter}) without converging. Their rho is not "
              "comparable to a layer that stopped on patience -- a partly "
              "fit factorization has not settled which feature carries what. "
              "Raise --max-iter or read those layers on their own.")
    run_metadata = {
        "model": a.model, "concept": a.concept, "k": k,
        "dtype": a.dtype, "n_layers": n_layers, "d_mlp": d_mlp,
        "n_concept_sentences": len(concept), "n_neutral_sentences": len(neutral),
        "sparsity": a.sparsity, "ridge": a.ridge, "tau": a.tau,
        "max_iter": a.max_iter, "rtol": a.rtol, "seed": a.seed,
        "activation_tokens_saved": a.top_tokens,
        "projection_tokens_saved": a.projection_tokens,
        "context_window": a.context_window,
        "max_per_token_type": a.max_per_token_type,
        "evidence_format": "token_context_score",
        "unconverged_layers": sorted(unconverged),
    }
    (out / "factorization_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    print("wrote", out / "features.pkl", "and both token-evidence files")


def check_both_sides_applied(applied_in, applied_out, delta_in, delta_out,
                            ranges, selected, cmd):
    """Refuse a one-sided erasure that the caller did not ask for.

    SNMF projects the selected directions out of BOTH up_proj (input side) and
    down_proj (output side). The two sides have separate layer ranges, so a
    layer can clear one and not the other -- and if every layer holding
    selected features falls outside a range, that side is silently never
    touched while `total` still counts the features from the other side. The
    total==0 guard cannot see it.

    This is what job 871388 produced: the judge accepted 0 of layer 4's 9
    candidates and 13 across layers 9 and 14, and since the output range on 18
    layers is [0,5], every selected feature sat outside it. The erase would
    have applied delta_in at two layers, delta_out nowhere, and saved a model
    labelled as an SNMF erasure. A deliberate --delta-out 0 is still allowed;
    only asking for a side and receiving nothing is an error.
    """
    (lo_in, hi_in), (lo_out, hi_out) = ranges
    have = sorted(int(l) for l, f in selected.items() if f)
    for name, applied, delta, (lo, hi), other in (
            ("output-side (down_proj)", applied_out, delta_out, (lo_out, hi_out),
             "--layers-out"),
            ("input-side (up_proj)", applied_in, delta_in, (lo_in, hi_in),
             "--layers-in")):
        if applied or not delta:
            continue
        raise SystemExit(
            f"{cmd}: --delta-{'out' if 'output' in name else 'in'}={delta} was "
            f"requested but the {name} edit reached NO layer, so the result "
            f"would be a one-sided erasure saved under the method's name.\n"
            f"  layers holding selected features: {have}\n"
            f"  {other} range in effect:          [{lo},{hi}]\n"
            f"They do not intersect. Either factorize a layer inside "
            f"[{lo},{hi}] and select features there, widen the range with "
            f"{other} lo hi, or pass the delta for this side as 0 to say the "
            f"one-sided edit is intended.")


def cmd_erase(a):
    model, tok = load_model(a.model, a.cache_dir, DTYPES[a.dtype],
                            resolve_device(a.device))
    blob = pickle.load(open(Path(a.out) / "features.pkl", "rb"))
    selected_path = Path(a.out) / "selected.json"
    if not selected_path.exists():
        raise SystemExit(
            "selected.json is missing. Run the select command first.")
    selected = json.load(open(selected_path))
    selection_meta_path = Path(a.out) / "selection_metadata.json"
    if not selection_meta_path.exists():
        raise SystemExit("selection_metadata.json is missing")
    selection_meta = json.load(open(selection_meta_path))
    print("selection method:", selection_meta.get("selection_method", "unknown"))

    (lo_in, hi_in), (lo_out, hi_out) = resolve_ranges(
        len(get_layers(model)), a.layers_in, a.layers_out, a.range_cell)
    total = 0
    applied_in = applied_out = 0
    for l in blob["layers"]:
        feats = selected.get(str(l), [])
        if not feats:
            continue
        d_in = a.delta_in if lo_in <= l <= hi_in else 0.0
        d_out = a.delta_out if lo_out <= l <= hi_out else 0.0
        if not (d_in or d_out):
            continue
        n, kept = ablate_layer(model, l, blob["results"][l]["Z"], feats,
                               d_in, d_out, a.gamma)
        total += n
        applied_in += bool(d_in)
        applied_out += bool(d_out)
        print(f"layer {l}: {n} features, {kept} neurons kept, "
              f"delta_in={d_in} delta_out={d_out}")
    print(f"{total} feature ablations total "
          f"({applied_in} layers edited input-side, "
          f"{applied_out} output-side)")
    if total == 0:
        # Saving here would write a model byte-identical to its control and
        # call it erased. Nothing downstream could tell.
        raise SystemExit(
            "no feature was ablated: selected.json is empty for every layer "
            f"in [{lo_in},{hi_in}] u [{lo_out},{hi_out}], or both deltas are "
            "zero. Nothing saved.")
    check_both_sides_applied(
        applied_in, applied_out, a.delta_in, a.delta_out,
        ((lo_in, hi_in), (lo_out, hi_out)), selected, "erase")
    if a.save:
        model.save_pretrained(a.save)
        tok.save_pretrained(a.save)
        erasure_meta = {
            "method": "SNMF",
            "selection": selection_meta,
            "delta_in": a.delta_in, "delta_out": a.delta_out,
            "layers_in": [lo_in, hi_in], "layers_out": [lo_out, hi_out],
            "layers_in_from": "cli" if a.layers_in else "model depth",
            "layers_out_from": "cli" if a.layers_out else "model depth",
            "dtype": a.dtype,
            "gamma_cov": a.gamma, "feature_ablations": total,
            "layers_edited_input_side": applied_in,
            "layers_edited_output_side": applied_out,
        }
        (Path(a.save) / "snmf_erasure_metadata.json").write_text(
            json.dumps(erasure_meta, indent=2))
        print("saved", a.save)


# Two STAGE1 prompts, one per evidence source, both ported from the upstream
# reference in external/snmf/experiments/snmf_interp/. The single bare-token
# prompt that used to serve both is what produced the weak descriptions:
# it gave the judge no context, no scores, and -- because the old evidence was
# token positions rather than types -- frequently the same string 20 times.
#
# STAGE1_ACTIVATION mirrors CONCEPT_PROMPT in generate_input_descriptions.py.
# Its point 2 is the branch that needs the contexts: when the high-scoring
# samples are mostly the same token, read the tokens; otherwise read around
# them. STAGE1_PROJECTION mirrors CONNECTION_PROMPT in
# generate_output_centric_descriptions.py, which has no contexts to give
# (vocabulary logits, not corpus positions) but does carry the TRASH escape,
# so an incoherent feature can be named as incoherent instead of having a
# description confabulated for it.

STAGE1_ACTIVATION = """You are given a set of tokens, their surrounding context (words before and after the token), and an importance score.
Your task is to determine what is the connection between all the tokens.

### Instructions:

1. **Focus on High-Importance Samples:**
   Examine only the token-context pairs with the highest importance scores. If a significant drop is observed beyond a threshold, ignore the lower-scoring pairs.

2. **Assess Token Consistency vs. Contextual Patterns:**
   - **Token Consistency:** If the high-importance samples are mostly identical tokens or strongly related tokens, then consider the tokens only as the primary contributor.
   - **Contextual Patterns:** If the tokens are not related to one another, then focus on common semantic, syntactic, or structural patterns in the surrounding contexts.

3. Always choose the simplest and most obvious underlying connection. If inspecting the tokens alone is enough to find a connection, do not mention or utilize the contexts.

4. If there is no concept or underlying connection between the tokens, output under the Results section exactly: TRASH

### Output Format:

Analysis:
<reason what the underlying connection is.>

Results:
<Single sentence description of the single most obvious connection between the tokens>

### Input:

Token-Context Pairs:
```{evidence}```

Remember, find the most obvious connection prioritizing connecting the tokens alone without the contexts and only if you cannot find any connection between the tokens then you may inspect their contexts."""

STAGE1_PROJECTION = """You are given a set of tokens and their importance score.
Your task is to determine what is the connection between all the tokens.

### Instructions:

1. **Focus on High-Importance Samples:**
   Examine only the token-score pairs with the highest importance scores. If a significant drop is observed beyond a threshold, ignore the lower-scoring tokens.

2. Always choose the simplest and most obvious underlying connection.

3. Ignore noisy tokens. Some tokens may be unrelated to the concept, only consider tokens that share the most obvious connection.

4. If there is no concept or underlying connection between the tokens, output under the Results section exactly: TRASH

### Output Format:

Analysis:
<reason about what the underlying connection is.>

Results:
<Single sentence description of the single most obvious connection between the tokens>

### Input:

Tokens:
```{evidence}```"""

# Retained so anything still importing STAGE1 keeps working; it is no longer
# used by cmd_select.
STAGE1 = STAGE1_PROJECTION

STAGE2 = """You get a concept name and a feature's description.
Decide if this feature describes the given concept. Consider if the feature is
DISTINCTIVE for the concept (not broad like "sport" for "basketball") and not
too noisy. A feature should be marked true only if the concept is central and
dominant in the description, not just one of several themes.
Return ONLY JSON: {{"is_member": true|false, "confidence": 0..1}}

Example 1:
Concept: Charity
Description: This vector represents elements related to giving and donations to charitable causes.
Answer: {{"is_member": true, "confidence": 0.98}}

Example 2:
Concept: Charity
Description: This vector represents elements related to money, such as gambling, investing and charity.
Answer: {{"is_member": false, "confidence": 0.80}}

Concept: {concept}
Description: {description}
Answer:"""


# Judge defaults are the ones the EMBER run on this same 1B used
# (configs/ember_lment_rome_slurm.yaml): the pinned revision, the 23 GB
# already in hf_cache, its own interpreter, and the 3-hour startup budget that
# an unsharded 23.9 GB safetensors read off this filer actually needs.
GEMMA_JUDGE = "google/gemma-4-12B-it"
GEMMA_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
GEMMA_CACHE = "/home/dcor/galbarak2/hf_cache/hub"
GEMMA_PYTHON = "/home/dcor/galbarak2/conda_envs/gemma/bin/python"
GEMMA_STARTUP_TIMEOUT = 10800.0


class LocalGemmaClient:
    """The project's own judge, standing in for Gemini.

    Reuses ``ember.subprocess_judge.SubprocessJudge`` from the EMBER fork on
    this branch. Its ``describe_feature`` / ``classify_feature`` seams take a
    raw prompt and return text, and ``classify_feature`` already normalises
    the reply to ``{"is_member": bool, "confidence": float}`` -- exactly what
    ``_parse_membership`` below expects -- so STAGE1/STAGE2 are unchanged and
    the only thing that differs from the published recipe is which model
    answers them.
    """

    def __init__(self, model=GEMMA_JUDGE, revision=GEMMA_REVISION,
                 python_executable=GEMMA_PYTHON, cache_dir=GEMMA_CACHE,
                 device="cuda", max_new_tokens=256,
                 startup_timeout=GEMMA_STARTUP_TIMEOUT, fork_root=None):
        root = Path(fork_root) if fork_root else (
            Path(__file__).resolve().parent / "Ember-on-LMEnt")
        if not (root / "ember").is_dir():
            raise SystemExit(
                f"--judge gemma needs the EMBER fork: no ember/ package under "
                f"{root}. Pass --fork-root, or run from a checkout of "
                "itamars/Ember-on-LMEnt.")
        sys.path.insert(0, str(root))
        from ember.subprocess_judge import (SubprocessJudge,
                                            resolve_judge_directory)
        model_path = resolve_judge_directory(
            model, revision=revision, cache_dir=cache_dir,
            local_files_only=True)
        print(f"judge: {model_path} via {python_executable}")
        self._judge = SubprocessJudge(
            model_path=model_path, python_executable=python_executable,
            device=device, max_new_tokens=max_new_tokens, cache_dir=cache_dir,
            local_files_only=True, project_root=root,
            startup_timeout=startup_timeout)
        self.metadata = {"judge_backend": "gemma-subprocess",
                         "judge_model": str(model), "judge_revision": revision,
                         "judge_path": str(model_path)}

    def describe(self, prompt):
        return self._judge.describe_feature(prompt)

    def classify(self, prompt):
        return self._judge.classify_feature(prompt)

    def close(self):
        self._judge.close()


class GeminiClient:
    def __init__(self, model_name, max_retries=3, sleep_seconds=5.0):
        key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_TOKEN")
               or os.getenv("GEMINI_API_KEY"))
        if not key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY, GEMINI_API_TOKEN, or GEMINI_API_KEY")
        try:
            import google.generativeai as gai
        except ImportError as exc:
            raise RuntimeError("Install google-generativeai") from exc
        gai.configure(api_key=key)
        self.model = gai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.metadata = {"judge_backend": "gemini", "judge_model": model_name}

    def describe(self, prompt):
        return self.generate(prompt)

    def classify(self, prompt):
        return self.generate(prompt)

    def close(self):
        return None

    def generate(self, prompt):
        last = None
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                text = getattr(response, "text", "")
                if not text or not text.strip():
                    raise RuntimeError("Gemini returned empty text")
                return text.strip()
            except Exception as exc:
                last = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.sleep_seconds)
        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last}")


def _parse_membership(text):
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in Gemini response: {text!r}")
    obj = json.loads(match.group(0))
    return bool(obj["is_member"]), float(obj["confidence"])


def _extract_results(text):
    """Pull the Results section out of the upstream Analysis/Results format.

    Falls back to the whole reply, so a judge that ignores the format still
    produces something usable rather than an empty description.
    """
    match = re.search(r"Results:\s*(.*)", text, flags=re.DOTALL)
    body = match.group(1) if match else text
    return body.strip().strip("`").strip()


def _render_evidence(source, data, top_k):
    """Format one feature's evidence for STAGE1, per source.

    `data` is the list stored by factorize. New runs store dicts with token /
    context / activation; older runs stored bare strings. Both are accepted so
    an existing out/ directory still judges, but the bare-string path loses
    the contexts and is reported as degraded by the caller.
    """
    rows = list(data)[: max(top_k * 4, top_k)]
    rich = bool(rows) and isinstance(rows[0], dict)
    if source == "activation":
        if rich:
            return format_context_evidence(rows, top_m=top_k), True
        # no contexts available: fall back to the projection-style rendering
        return "\n".join(f"Token: `{t}`" for t in rows[:top_k]), False
    if rich:
        rows = sorted(rows, key=lambda d: -float(d.get("score", 0.0)))[:top_k]
        return "\n".join(
            f"Token: `{d['token']}` | Score: `{d.get('score', 0.0)}`"
            for d in rows), True
    return "\n".join(f"Token: `{t}`" for t in rows[:top_k]), False


def _judge_evidence(client, concept, data, top_k, confidence_threshold,
                    source="activation"):
    evidence, rich = _render_evidence(source, data, top_k)
    template = STAGE1_ACTIVATION if source == "activation" else STAGE1_PROJECTION
    raw_stage1 = client.describe(template.format(evidence=evidence))
    description = _extract_results(raw_stage1)

    # TRASH is the upstream escape for "these tokens share no concept". Sending
    # it to STAGE2 asks the judge whether the string TRASH describes Ancient
    # Rome, which is not a question about the feature.
    if description.upper().startswith("TRASH") or not description:
        return {"source": source, "evidence": evidence, "has_context": rich,
                "description": description or "TRASH", "is_member": False,
                "confidence": 1.0, "accepted": False, "trash": True,
                "raw_stage1": raw_stage1, "raw_stage2": None}

    raw = client.classify(STAGE2.format(concept=concept, description=description))
    member, confidence = _parse_membership(raw)
    return {
        "source": source, "evidence": evidence, "has_context": rich,
        "description": description, "is_member": member,
        "confidence": confidence,
        "accepted": member and confidence >= confidence_threshold,
        "trash": False, "raw_stage1": raw_stage1, "raw_stage2": raw,
    }


def cmd_select(a):
    """Select the concept features used by the weight update."""
    out = Path(a.out)
    candidates = json.load(open(out / "candidates.json"))

    # --skip-llm keeps every rho > tau candidate instead of running the
    # Gemini filter. Factorization and ablation are unchanged; only this
    # selection step stops matching Appendix A.3.
    if a.skip_llm:
        selected = {
            layer: [int(feature) for feature in features]
            for layer, features in candidates.items()
        }
        (out / "selected.json").write_text(json.dumps(selected, indent=2))
        metadata = {
            "selection_method": "ratio_only",
            "llm_filter_used": False,
            "tau": json.load(open(out / "factorization_metadata.json"))["tau"],
            "selected_count": sum(map(len, selected.values())),
        }
        (out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"selected {metadata['selected_count']} ratio-filtered features")
        return

    if not a.concept:
        raise SystemExit("--concept is needed unless --skip-llm is used")

    activating = json.load(open(out / "feature_tokens.json"))
    projecting_path = out / "projection_tokens.json"
    if not projecting_path.exists():
        raise SystemExit(
            "projection_tokens.json is missing; rerun factorize with this version. "
            "Both token sources are needed for feature selection.")
    projecting = json.load(open(projecting_path))
    if sum(map(len, candidates.values())) == 0:
        raise SystemExit(
            "no candidate features to judge: every layer's rho fell short of "
            "tau at factorization time. Read out/rho_stats.json before "
            "changing tau, and write down the rule you used.")
    if a.judge == "gemini":
        client = GeminiClient(a.gemini_model, a.max_retries, a.retry_sleep)
    else:
        client = LocalGemmaClient(
            model=a.judge_model, revision=a.judge_revision,
            python_executable=a.judge_python, cache_dir=a.judge_cache_dir,
            device=a.judge_device, max_new_tokens=a.judge_max_new_tokens,
            startup_timeout=a.judge_startup_timeout, fork_root=a.fork_root)
    selected = {layer: [] for layer in candidates}
    records = []
    degraded = 0
    interpretations_path = out / "interpretations.json"
    try:
        for layer, ids in candidates.items():
            for fid_value in ids:
                fid = str(fid_value)
                record = {"layer": int(layer), "feature": int(fid)}
                for source, data in (("activation", activating),
                                     ("projection", projecting)):
                    print(f"judge layer={layer} feature={fid} source={source}",
                          flush=True)
                    record[source] = _judge_evidence(
                        client, a.concept, data[layer][fid],
                        a.judge_top_tokens, a.confidence_threshold,
                        source=source)
                if not record["activation"]["has_context"]:
                    degraded += 1
                if (record["activation"]["accepted"]
                        or record["projection"]["accepted"]):
                    selected[layer].append(int(fid))
                records.append(record)
                interpretations_path.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False))
    finally:
        client.close()
    if degraded:
        print(f"\nWARNING: {degraded} of {len(records)} features were judged "
              "from bare token strings with no context, because "
              "feature_tokens.json came from a factorize run that predates "
              "context capture. That is the evidence format the judge "
              "performs worst on. Re-run factorize to restore it.")
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    metadata = {
        "selection_method": f"ratio_then_{a.judge}",
        "llm_filter_used": True,
        "confidence_threshold": a.confidence_threshold,
        "judge_top_tokens": a.judge_top_tokens,
        "sources": ["activation", "projection"],
        "union_rule": "accepted by either independently judged source",
        "selected_count": sum(map(len, selected.values())),
        "n_candidates_judged": len(records),
    }
    metadata.update(client.metadata)
    (out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"selected {metadata['selected_count']} of {len(records)} candidates "
          f"({a.judge} judge)")


@torch.no_grad()
def feature_activation(acts, Z, feature_ids):
    """Mean projection of activations onto each concept feature direction."""
    if not feature_ids:
        return torch.zeros(0)
    Zs = Z[:, feature_ids].to(acts.dtype)
    Zs = Zs / Zs.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return (Zs.T @ acts).abs().mean(dim=1)          # (n_features,)


@torch.no_grad()
def feature_directions(model, layer_id, Z, feature_ids):
    """Unit readout direction f_i = W_out^T z_i, from the weights AS THEY ARE.

    Capture these before an edit and hand them to feature_readout afterwards.
    Recomputing the direction from edited weights does not measure the edit --
    see the warning in feature_readout.
    """
    W_out = mlp_of(get_layers(model)[layer_id]).down_proj.weight.data.T
    dirs = {}
    for i in feature_ids:
        z = Z[:, i].to(W_out.device, W_out.dtype)
        if not (z != 0).any():
            continue
        f = W_out.T @ z
        dirs[int(i)] = (f / f.norm().clamp(min=1e-8)).clone()
    return dirs


def feature_readout(model, layer_id, Z, feature_ids, directions=None):
    """How much of each feature's direction survives in down_proj.

    The activation metric above cannot see the output-side edit AT ALL.
    collect_activations hooks the *input* to down_proj, and editing down_proj
    does not change its own input -- only what the layer writes to the
    residual stream, which shows up at the next layer if anywhere. Every
    number in the verify table was therefore reporting the up_proj edit alone,
    including for layers where --delta-out was the only thing applied.

    This measures the other side directly: the norm of W_out restricted to the
    feature's support, projected on the feature's readout direction. It is a
    weight statistic, not an activation one, so it needs no forward pass.

    `directions` MUST be supplied for the post-edit measurement, and must come
    from feature_directions() called on the unedited weights.

    Why: ablate_layer subtracts delta * outer(W_out f, f) on the support, so
    afterwards W_out^T z = u - delta * (u.f_hat) f_hat = (1 - delta) * u. At
    delta = 1 -- exact removal, and one of the four values in the published
    sweep {1, 4, 7, 10} -- that is analytically ZERO, and recomputing the
    direction from the edited matrix divides cancellation residue by its own
    norm. Measured on a 6-layer Olmo2: ||W_out^T z|| collapses 2.58e-1 ->
    3.92e-8 exactly as intended, but the recomputed-direction readout reports
    0.0331 -> 0.0088, a 73% drop made entirely of rounding noise. In float64
    the residue falls under the 1e-8 clamp instead and the same call reports
    100%. Against the fixed direction it reads 0.0331 -> 4.97e-9, a clean
    1.5e-7 collapse, in either dtype.

    At delta != 1 the recomputed direction stays parallel to the original and
    .abs() hides the sign, so the old path happened to agree there. delta = 1
    is precisely the cell it got wrong.
    """
    if not feature_ids:
        return torch.zeros(0)
    W_out = mlp_of(get_layers(model)[layer_id]).down_proj.weight.data.T   # (d_mlp, d)
    vals = []
    for i in feature_ids:
        z = Z[:, i].to(W_out.device, W_out.dtype)
        support = (z != 0)
        if not support.any():
            vals.append(torch.zeros((), device=W_out.device, dtype=W_out.dtype))
            continue
        if directions is not None and int(i) in directions:
            f = directions[int(i)].to(W_out.device, W_out.dtype)
        else:
            f = W_out.T @ z
            f = f / f.norm().clamp(min=1e-8)
        vals.append((W_out[support] @ f).abs().mean())
    return torch.stack(vals).float().cpu()


def cmd_verify(a):
    """Compare feature activation before and after the weight update."""
    blob = pickle.load(open(Path(a.out) / "features.pkl", "rb"))
    sel_path = Path(a.out) / "selected.json"
    if not sel_path.exists():
        raise SystemExit("selected.json is missing; choose a feature-selection method first")
    selected = json.load(open(sel_path))
    layers = [l for l in blob["layers"] if selected.get(str(l))]
    if not layers:
        raise SystemExit("no selected features in any layer")

    model, tok = load_model(a.model, a.cache_dir, DTYPES[a.dtype],
                            resolve_device(a.device))
    concept = load_sentences(a.concept_sentences, a.concept)[:a.n_sentences]
    neutral = load_sentences(a.neutral_sentences)[:a.n_sentences]

    def measure():
        acts, is_c, _, _ = collect_activations(model, tok, concept, neutral,
                                               layers, a.batch_size, a.max_length)
        return {l: (feature_activation(acts[l][:, is_c], blob["results"][l]["Z"],
                                       selected[str(l)]).mean().item(),
                    feature_activation(acts[l][:, ~is_c], blob["results"][l]["Z"],
                                       selected[str(l)]).mean().item())
                for l in layers}

    # Directions are taken ONCE, from the unedited weights, and reused for the
    # after measurement. Recomputing them post-edit measures nothing at
    # delta=1, where W_out^T z is analytically zero -- see feature_readout.
    out_dirs = {l: feature_directions(model, l, blob["results"][l]["Z"],
                                      selected[str(l)])
                for l in layers}

    def measure_out():
        return {l: feature_readout(model, l, blob["results"][l]["Z"],
                                   selected[str(l)], out_dirs[l]).mean().item()
                for l in layers}

    before = measure()
    before_out = measure_out()
    (lo_in, hi_in), (lo_out, hi_out) = resolve_ranges(
        len(get_layers(model)), a.layers_in, a.layers_out, a.range_cell)
    applied_in = applied_out = 0
    for l in layers:
        d_in = a.delta_in if lo_in <= l <= hi_in else 0.0
        d_out = a.delta_out if lo_out <= l <= hi_out else 0.0
        if d_in or d_out:
            ablate_layer(model, l, blob["results"][l]["Z"], selected[str(l)],
                         d_in, d_out, a.gamma)
            applied_in += bool(d_in)
            applied_out += bool(d_out)
    check_both_sides_applied(
        applied_in, applied_out, a.delta_in, a.delta_out,
        ((lo_in, hi_in), (lo_out, hi_out)),
        {str(l): selected[str(l)] for l in layers}, "verify")
    after = measure()
    after_out = measure_out()

    print("\nINPUT SIDE (up_proj). These are down_proj *input* activations, so")
    print("they respond to the up_proj edit at this layer and to anything")
    print("edited upstream. A down_proj edit at the same layer is invisible")
    print("here by construction -- see the output-side table below.")
    print(f"{'layer':>6} {'concept before':>15} {'after':>10} {'drop':>7} "
          f"{'neutral before':>15} {'after':>10} {'drop':>7}")
    drops_c, drops_n = [], []
    for l in layers:
        cb, nb = before[l]
        ca, na = after[l]
        dc = 1 - ca / max(cb, 1e-12)
        dn = 1 - na / max(nb, 1e-12)
        drops_c.append(dc)
        drops_n.append(dn)
        print(f"{l:>6} {cb:>15.4g} {ca:>10.4g} {dc:>6.1%} "
              f"{nb:>15.4g} {na:>10.4g} {dn:>6.1%}")
    mc = sum(drops_c) / len(drops_c)
    mn = sum(drops_n) / len(drops_n)
    print(f"\nmean concept drop {mc:.1%}, mean neutral drop {mn:.1%}, "
          f"selectivity {mc - mn:+.1%}")

    print("\nOUTPUT SIDE (down_proj). Weight statistic, no forward pass: mean "
          "|W_out . f| over\nthe feature's support. This is the only column "
          "that moves when --delta-out is the\nonly thing applied at a layer.")
    print(f"{'layer':>6} {'readout before':>15} {'after':>10} {'drop':>7}")
    drops_o = []
    for l in layers:
        ob, oa = before_out[l], after_out[l]
        do = 1 - oa / max(ob, 1e-12)
        drops_o.append(do)
        print(f"{l:>6} {ob:>15.4g} {oa:>10.4g} {do:>6.1%}")
    mo = sum(drops_o) / len(drops_o)
    print(f"\nmean output-side drop {mo:.1%}")
    if applied_out == 0:
        print("  (no layer received --delta-out, so any movement here is "
              "numerical noise)")
    # The update is (I - delta * P) on the feature's support, so the component
    # along the feature scales by |1 - delta| and this metric, which takes
    # .abs() of that component, reports a drop of 1 - |1 - delta|:
    #
    #     delta 1 -> +100% (exact removal)   delta 2 -> 0%
    #     delta 4 -> -200%   delta 7 -> -500%   delta 10 -> -800%
    #
    # Verified numerically against the reference update in mlp_edit.intervene.
    #
    # delta 4 is NOT "the published value". configs/snmf_gemma.yaml sets
    #   in_deltas:  [1.0, 4.0, 7.0, 10.0]
    #   out_deltas: [1.0, 4.0, 7.0, 10.0]
    # and SNMFMethod.enumerate_hps crosses every one of them with every layer
    # range. The reference picks a cell on downstream evaluation --
    # `max_qa_acc: 0.6` on the concept questions and `min_mmlu: 0.7` for
    # collateral damage -- and never on an activation drop. So there is no
    # inconsistency to resolve between delta and this metric: this metric was
    # never the selection criterion. Use it to confirm the edit lands where
    # it should, then sweep delta and read the task numbers.
    predicted = 1.0 - abs(1.0 - max(a.delta_in, a.delta_out))
    if mc < 0 or mn < 0:
        print(f"\nNOTE: activation went UP, and at delta="
              f"{max(a.delta_in, a.delta_out)} that is expected, not a fault. "
              f"The update is (I - delta*P), so the component scales by "
              f"|1 - delta| = {abs(1.0 - max(a.delta_in, a.delta_out)):.2f} and "
              f"this metric predicts a drop of {predicted:+.0%} "
              f"(observed {mc:+.1%} concept).")
        print("  Do NOT raise delta to fix this; it moves further from zero.")
        print("  delta < 2 shrinks the component, delta = 1 removes it exactly,")
        print("  delta = 2 flips its sign leaving the magnitude untouched.")
        print("  The reference sweeps delta over {1, 4, 7, 10} and selects on "
              "downstream\n  task accuracy, not on this number. Treat the "
              "table above as a check that\n  the edit reached the right "
              "matrices, and choose delta with an eval.")
    elif mc < 0.2:
        print("concept drop is small; the edit is not doing much. Check the "
              "up_proj/down_proj transposes before changing delta.")
    if mc > 0 and mn > mc * 0.7:
        print("neutral drops nearly as much as concept; the features are not "
              "concept-specific. Raise tau or apply the LLM filter.")
    elif mc < 0 and mn < 0:
        print("  Neutral moved the same way for the same reason -- the scaling "
              "applies to everything in the feature's support, so this says "
              "nothing about specificity either way.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("factorize")
    f.add_argument("--model", required=True)
    f.add_argument("--concept", required=True)
    f.add_argument("--concept-sentences", required=True)
    f.add_argument("--neutral-sentences", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--cache-dir")
    f.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    f.add_argument("--device", help="default cuda:0, or cpu without CUDA")
    f.add_argument("--layers", type=int, nargs="+")
    f.add_argument("--k", type=int)
    f.add_argument("--n-sentences", type=int, default=300)
    f.add_argument("--allow-short-data", action="store_true")
    f.add_argument("--sparsity", type=float, default=0.01)
    # external/snmf/factorization/seminmf.py fit() takes reg=1e-6; train.py
    # never overrides it, so 1e-6 is the reference value, not 1e-4.
    f.add_argument("--ridge", type=float, default=1e-6)
    f.add_argument("--tau", type=float, default=2.0)
    f.add_argument("--max-iter", type=int, default=20000)
    f.add_argument("--rtol", type=float, default=1e-6,
                   help="relative improvement that counts as progress. The "
                        "old absolute tol=1e-4 was meaningless against a "
                        "reconstruction error of 1e12 and let layers stop at "
                        "different degrees of convergence.")
    f.add_argument("--batch-size", type=int, default=8)
    f.add_argument("--layer-batch-size", type=int, default=1)
    f.add_argument("--max-length", type=int, default=256)
    f.add_argument("--top-tokens", type=int, default=25,
                   help="activating examples kept per feature; upstream's "
                        "generate_concept_context.py uses 25")
    f.add_argument("--projection-tokens", type=int, default=30)
    f.add_argument("--context-window", type=int, default=15,
                   help="tokens either side of an activating token, clipped "
                        "at the sentence boundary (upstream default 15)")
    f.add_argument("--max-per-token-type", type=int, default=3,
                   help="cap on instances of one token string in a feature's "
                        "evidence. Without a cap the top-N by coefficient is "
                        "frequently ONE string repeated N times, because Y "
                        "is indexed by token position and not by type.")
    f.add_argument("--seed", type=int, default=42)
    f.set_defaults(func=cmd_factorize)

    e = sub.add_parser("erase")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--cache-dir")
    e.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    e.add_argument("--device", help="default cuda:0, or cpu without CUDA")
    e.add_argument("--delta-in", type=float, default=4.0)
    e.add_argument("--delta-out", type=float, default=4.0)
    # default None, resolved from the model's depth; see default_layer_ranges
    e.add_argument("--layers-in", type=int, nargs=2)
    e.add_argument("--layers-out", type=int, nargs=2)
    e.add_argument("--range-cell", type=int, default=0, choices=(0, 1, 2),
                   help="which cell of the published layer-range grid to use. "
                        "The reference sweeps three per side; cell 0 is the "
                        "only one that was reachable before. On 18 layers "
                        "cell 2 gives out=[9,17], which is where the judge's "
                        "features actually live.")
    e.add_argument("--gamma", type=float, default=0.95)
    e.add_argument("--save")
    e.set_defaults(func=cmd_erase)

    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--concept")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--judge", choices=("gemma", "gemini"), default="gemma",
                   help="gemma is the local pinned judge this project already "
                        "uses for embedding features; gemini needs a key")
    p.add_argument("--judge-model", default=GEMMA_JUDGE)
    p.add_argument("--judge-revision", default=GEMMA_REVISION)
    p.add_argument("--judge-python", default=GEMMA_PYTHON)
    p.add_argument("--judge-cache-dir", default=GEMMA_CACHE)
    p.add_argument("--judge-device", default="cuda")
    p.add_argument("--judge-max-new-tokens", type=int, default=256)
    p.add_argument("--judge-startup-timeout", type=float,
                   default=GEMMA_STARTUP_TIMEOUT)
    p.add_argument("--fork-root",
                   help="path to the Ember-on-LMEnt checkout holding ember/")
    p.add_argument("--gemini-model", default="models/gemini-2.5-flash-lite")
    p.add_argument("--judge-top-tokens", type=int, default=10,
                   help="examples actually shown to STAGE1. Upstream's "
                        "generate_input_descriptions.py uses --top-m 10.")
    p.add_argument("--confidence-threshold", type=float, default=0.85)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=5.0)
    p.set_defaults(func=cmd_select)

    v = sub.add_parser("verify")
    v.add_argument("--model", required=True)
    v.add_argument("--concept", required=True)
    v.add_argument("--concept-sentences", required=True)
    v.add_argument("--neutral-sentences", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--cache-dir")
    v.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    v.add_argument("--device", help="default cuda:0, or cpu without CUDA")
    v.add_argument("--delta-in", type=float, default=4.0)
    v.add_argument("--delta-out", type=float, default=4.0)
    v.add_argument("--layers-in", type=int, nargs=2)
    v.add_argument("--layers-out", type=int, nargs=2)
    v.add_argument("--range-cell", type=int, default=0, choices=(0, 1, 2),
                   help="which cell of the published layer-range grid to use. "
                        "The reference sweeps three per side; cell 0 is the "
                        "only one that was reachable before. On 18 layers "
                        "cell 2 gives out=[9,17], which is where the judge's "
                        "features actually live.")
    v.add_argument("--gamma", type=float, default=0.95)
    v.add_argument("--n-sentences", type=int, default=300)
    v.add_argument("--batch-size", type=int, default=8)
    v.add_argument("--max-length", type=int, default=256)
    v.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
