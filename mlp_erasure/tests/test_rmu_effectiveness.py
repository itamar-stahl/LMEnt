"""Tests for why RMU produced no measurable erasure on the LMEnt 1B.

Two causes, both of which let a run report success:

  1. The step count was capped by the RETAIN pool. RMU's retain term is a
     regulariser -- the reference draws it from wikitext, which never runs out
     -- but neutral_sentences.json holds 300 sentences TOTAL for all concepts,
     and `n = min(max_num_batches, forget, retain)` let that file decide how
     much unlearning happened.

  2. The sanity gate's bar was `cos_forget_end > cos_forget_start + 0.05`.
     Job 871273 went 0.0173 -> 0.0782 and reported five green booleans, but
     0.08 cosine means the forget representations are still essentially
     ORTHOGONAL to the control vector they were supposed to be pointed at.

    python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import rmu  # noqa: E402

DATA = ROOT / "Ember-on-LMEnt" / "data"


def trace_with(cos_forget, cos_retain=None, unlearn=None):
    n = len(cos_forget)
    return {
        "cos_forget": list(cos_forget),
        "cos_retain": list(cos_retain if cos_retain is not None else [0.999] * n),
        "unlearn": list(unlearn if unlearn is not None else
                        [10.0 - i for i in range(n)]),
        "retain": [0.0] * n,
    }


class StepCountIsNotCappedByRetain(unittest.TestCase):
    """The retain pool must not decide how much unlearning happens."""

    def test_retain_indexing_wraps(self):
        # 3 retain batches over 7 steps must not IndexError and must cycle
        retain = ["r0", "r1", "r2"]
        seen = [retain[i % len(retain)] for i in range(7)]
        self.assertEqual(seen, ["r0", "r1", "r2", "r0", "r1", "r2", "r0"])

    def test_build_batches_applies_both_length_bounds(self):
        forget = ["x" * 10, "y" * 100, "z" * 3000]
        retain = ["a" * 80, "b" * 20]
        fb, rb = rmu.build_batches(forget, retain, batch_size=1,
                                   min_len=50, max_len=2000)
        self.assertEqual(sum(len(b) for b in fb), 1)   # only the 100-char one
        self.assertEqual(sum(len(b) for b in rb), 1)   # only the 80-char one

    def test_real_data_would_have_been_capped_by_the_neutral_file(self):
        """The measurement, on the files actually shipped."""
        concept = json.loads((DATA / "concept_sentences.json").read_text())
        rome = next(e for e in concept if "Rome" in str(e.get("concept", "")))
        neutral_raw = json.loads((DATA / "neutral_sentences.json").read_text())
        neutral = [(o.get("sentence") if isinstance(o, dict) else o)
                   for o in neutral_raw]

        fb, rb = rmu.build_batches([s for s in rome["sentences"] if s],
                                   [s for s in neutral if s],
                                   batch_size=4, min_len=50, max_len=2000)
        # retain is the smaller pool: this is the cap that used to bind
        self.assertLess(len(rb), len(fb))
        old_n = min(150, len(fb), len(rb))
        new_n = min(150, len(fb))
        self.assertGreater(new_n, old_n,
                           "cycling retain must free up steps the neutral "
                           "file was holding back")

    def test_neither_pool_reaches_the_published_schedule(self):
        """Honest bound: even uncapped, 300 sentences do not give 150 steps."""
        concept = json.loads((DATA / "concept_sentences.json").read_text())
        rome = next(e for e in concept if "Rome" in str(e.get("concept", "")))
        fb, _ = rmu.build_batches([s for s in rome["sentences"] if s],
                                  ["x" * 100] * 1000,
                                  batch_size=4, min_len=50, max_len=2000)
        self.assertLess(len(fb), 150,
                        "if this ever passes, the concept corpus grew and the "
                        "step-count limitation is gone -- update the docs")


class SanityGateRejectsAWeakRun(unittest.TestCase):
    """Reproduces job 871273's numbers and requires them to fail now."""

    ZERO = __import__("torch").zeros(4, 4)
    ONE = __import__("torch").ones(4, 4)

    def _sanity(self, cos_forget, **kw):
        import torch
        kw.setdefault("steps", len(cos_forget))
        kw.setdefault("requested_steps", len(cos_forget))
        return rmu.sanity(trace_with(cos_forget),
                          spare_before=torch.zeros(4, 4),
                          spare_after=torch.zeros(4, 4),
                          edited_before=torch.ones(4, 4),
                          edited_after=torch.ones(4, 4) * 1.01,
                          **kw)

    def test_job_871273_numbers_no_longer_count_as_success(self):
        # cos_forget 0.0173 -> 0.0782: cleared the old bar by 0.011
        res = self._sanity([0.0173] * 5 + [0.0782] * 5)
        self.assertTrue(res["forget_rotated"], "the old gate still passes")
        self.assertFalse(res["rotation_is_substantial"],
                         "but the run must not be reported as an erasure")
        failed = [k for k in rmu.SANITY_CHECKS if not res[k]]
        self.assertIn("rotation_is_substantial", failed)

    def test_a_real_misdirection_passes(self):
        res = self._sanity([0.02] * 5 + [0.85] * 5)
        self.assertTrue(res["rotation_is_substantial"])
        self.assertEqual([k for k in rmu.SANITY_CHECKS if not res[k]], [])

    def test_short_run_is_flagged(self):
        res = self._sanity([0.02] * 5 + [0.9] * 5,
                           steps=65, requested_steps=150)
        self.assertFalse(res["ran_enough_steps"])
        self.assertAlmostEqual(res["step_fraction"], 0.433, places=3)

    def test_full_schedule_passes_the_step_gate(self):
        res = self._sanity([0.02] * 5 + [0.9] * 5,
                           steps=150, requested_steps=150)
        self.assertTrue(res["ran_enough_steps"])

    def test_effect_size_is_reported_not_just_a_boolean(self):
        res = self._sanity([0.0173] * 5 + [0.0782] * 5)
        for key in ("cos_forget_end", "min_cosine_required", "step_fraction",
                    "edited_rel_change", "steps"):
            self.assertIn(key, res)

    def test_thresholds_are_configurable(self):
        res = self._sanity([0.02] * 5 + [0.10] * 5, min_cosine=0.05)
        self.assertTrue(res["rotation_is_substantial"])


class PublishedRMUGridIsASweep(unittest.TestCase):
    """Appendix C.3 tunes RMU; it does not publish one setting.

    The LMEnt run froze a single cell -- lr 1e-4, alpha 100, steering 100 --
    and alpha 100 is the TOP of the Gemma grid, i.e. the strongest retain
    penalty and so the least erasure available.
    """

    def setUp(self):
        sys.path.insert(0, str(ROOT / "Ember-on-LMEnt"))
        sys.path.insert(0, str(ROOT / "Ember-on-LMEnt" / "external" / "snmf"))
        try:
            from ember.erasure.methods import rmu as fork_rmu
            from ember.erasure.config import RunConfig
        except Exception as exc:                      # pragma: no cover
            self.skipTest(f"fork not importable: {exc}")
        self.fork_rmu = fork_rmu
        self.cfg = RunConfig()
        self.cfg.model_name = "google/gemma-2-2b-it"

    def test_the_grid_has_many_cells(self):
        lrs, alphas, steerings, settings = self.fork_rmu._grids_and_settings(self.cfg)
        self.assertEqual(len(lrs) * len(alphas) * len(steerings) * len(settings),
                         144)

    def test_alpha_100_is_the_most_conservative_cell(self):
        _, alphas, _, _ = self.fork_rmu._grids_and_settings(self.cfg)
        # higher alpha = heavier retain penalty = less unlearning
        self.assertEqual(max(alphas), 100.0)


if __name__ == "__main__":
    unittest.main()
