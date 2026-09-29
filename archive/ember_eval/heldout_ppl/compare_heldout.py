"""Compare two twins' loss on the held-out chunks. No GPU, no model.

The quantity of interest is the difference of differences:

    (ablated_heldout - control_heldout) - (ablated_ctrlset - control_ctrlset)

Each twin is compared against itself on chunks that were NOT masked, so any
global difference between the two models cancels and what remains is specific
to the 2,546 chunks the ablated twin never received gradient from.

Both twins score the SAME chunk ids, so the held-out comparison is paired
per chunk. A sign-flip permutation gives a p-value that assumes nothing about
the shape of the per-chunk differences -- the same test aggregate_completion.py
uses, for the same reason.

    python compare_heldout.py --control <json> --ablated <json>
"""
import argparse, json, random, statistics as st


def rows_by_id(blob, key):
    return {r["chunk_id"]: r for r in blob[key]}


def signflip_p(diffs, draws=20000, seed=42):
    if not diffs:
        return float("nan")
    rng = random.Random(seed)
    obs = abs(st.fmean(diffs))
    hits = 0
    for _ in range(draws):
        s = st.fmean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(s) >= obs:
            hits += 1
    return (hits + 1) / (draws + 1)


def paired(a, b, label, draws):
    """b - a over the chunk ids both scored."""
    common = sorted(set(a) & set(b))
    d = [b[i]["loss"] - a[i]["loss"] for i in common]
    if not d:
        return None
    mean = st.fmean(d)
    sd = st.pstdev(d) if len(d) > 1 else 0.0
    dz = mean / sd if sd else 0.0
    p = signflip_p(d, draws)
    print(f"  {label:22s} n={len(d):>5}  control={st.fmean([a[i]['loss'] for i in common]):.4f}  "
          f"ablated={st.fmean([b[i]['loss'] for i in common]):.4f}  "
          f"diff={mean:+.4f}  dz={dz:+.3f}  p={p:.4f}")
    return {"n": len(d), "mean_diff": mean, "dz": dz, "p": p, "diffs": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--ablated", required=True)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--out")
    a = ap.parse_args()

    C, A = json.load(open(a.control)), json.load(open(a.ablated))
    print(f"control model: {C['metadata']['model']}")
    print(f"ablated model: {A['metadata']['model']}\n")

    for name, blob in (("control", C), ("ablated", A)):
        h, c = blob["heldout"], blob["control"]
        print(f"{name:8s} held-out loss {h['mean_loss_micro']:.4f} (ppl {h['ppl_micro']:.3f}, "
              f"{h['n_tokens']} tok)   control-set loss {c['mean_loss_micro']:.4f} "
              f"(ppl {c['ppl_micro']:.3f}, {c['n_tokens']} tok)")

    print("\nper-chunk paired differences (ablated - control; POSITIVE = ablated "
          "finds it harder, which is the predicted direction on held-out):")
    held = paired(rows_by_id(C, "heldout_rows"), rows_by_id(A, "heldout_rows"),
                  "held-out chunks", a.draws)
    ctrl = paired(rows_by_id(C, "control_rows"), rows_by_id(A, "control_rows"),
                  "control chunks", a.draws)

    if held and ctrl:
        dd = held["mean_diff"] - ctrl["mean_diff"]
        print(f"\n  DIFFERENCE OF DIFFERENCES: {dd:+.4f} nats/token")
        print("  (held-out gap minus control gap; the control gap absorbs any")
        print("   global difference between the two models)")
        # unpaired across the two sets, so permute set labels
        rng = random.Random(7)
        pool = held["diffs"] + ctrl["diffs"]
        nh = len(held["diffs"])
        obs, hits = abs(dd), 0
        for _ in range(a.draws):
            rng.shuffle(pool)
            s = st.fmean(pool[:nh]) - st.fmean(pool[nh:])
            if abs(s) >= obs:
                hits += 1
        p_dd = (hits + 1) / (a.draws + 1)
        print(f"  label-permutation p = {p_dd:.4f}")

        if a.out:
            json.dump({"heldout": {k: v for k, v in held.items() if k != "diffs"},
                       "control": {k: v for k, v in ctrl.items() if k != "diffs"},
                       "diff_of_diffs": dd, "p": p_dd}, open(a.out, "w"), indent=2)
            print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
