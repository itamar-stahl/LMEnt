"""Tests for the feature-evidence, rho-scale and convergence fixes.

These pin the reasons the selection step was producing weak features. They
need torch but no checkpoint and no GPU, same as test_mlp_erasure.py.

    python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v
"""
import sys
import unittest
from pathlib import Path

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snmf   # noqa: E402


def tiny_olmo2(n_layers=6, hidden=32, mlp=88):
    """Same throwaway model test_mlp_erasure.py uses. No checkpoint, no GPU."""
    torch.manual_seed(0)
    return Olmo2ForCausalLM(Olmo2Config(
        vocab_size=64, hidden_size=hidden, intermediate_size=mlp,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=32,
        tie_word_embeddings=False)).eval()


class EvidenceIsTypesNotPositions(unittest.TestCase):
    """Y is indexed by token POSITION. The old evidence was not deduplicated,
    so a feature firing on a few token types handed the judge the same string
    over and over."""

    def setUp(self):
        # 200 positions across 4 sentences, drawn from a tiny vocabulary where
        # one string dominates -- the corpus shape that breaks the old path
        self.tokens = (["Rome"] * 60 + ["the"] * 60 + ["Caesar"] * 40
                       + ["legion"] * 40)
        self.sids = torch.tensor([i // 50 for i in range(200)])
        torch.manual_seed(0)
        self.Y = torch.rand(1, 200) * 0.01
        self.Y[0, :60] += 1.0                      # feature loves "Rome"

    def test_old_path_returns_one_string_many_times(self):
        old = snmf.top_tokens_per_feature(self.Y, self.tokens, top=20)[0]
        self.assertEqual(len(old), 20)
        self.assertEqual(len(set(old)), 1, "this is the bug being fixed")

    def test_dedup_caps_instances_of_one_string(self):
        got = snmf.top_contexts_per_feature(
            self.Y, self.tokens, self.sids, top=20, max_per_type=3)[0]
        counts = {}
        for item in got:
            counts[item["token"]] = counts.get(item["token"], 0) + 1
        self.assertTrue(all(c <= 3 for c in counts.values()), counts)
        self.assertGreater(len(counts), 1)

    def test_every_example_carries_context_and_score(self):
        got = snmf.top_contexts_per_feature(
            self.Y, self.tokens, self.sids, top=10)[0]
        self.assertTrue(got)
        for item in got:
            self.assertIn("token", item)
            self.assertIn("context", item)
            self.assertIn("activation", item)
            self.assertTrue(item["context"])

    def test_context_never_crosses_a_sentence_boundary(self):
        tokens = ["a", "b", "c", "d", "e", "f"]
        sids = torch.tensor([0, 0, 0, 1, 1, 1])
        Y = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])   # peak at index 2
        got = snmf.top_contexts_per_feature(
            Y, tokens, sids, top=1, context_window=5)[0]
        self.assertEqual(got[0]["token"], "c")
        self.assertEqual(got[0]["context"], "abc",
                         "window must stop at the end of sentence 0")

    def test_context_window_is_respected(self):
        tokens = [str(i) for i in range(21)]
        sids = torch.zeros(21, dtype=torch.long)
        Y = torch.zeros(1, 21)
        Y[0, 10] = 1.0
        got = snmf.top_contexts_per_feature(
            Y, tokens, sids, top=1, context_window=2)[0]
        self.assertEqual(got[0]["context"], "89101112")


class EvidenceRendering(unittest.TestCase):
    def test_activation_source_renders_token_context_score(self):
        items = [{"token": "Rome", "context": "in Rome the", "activation": 0.9}]
        text, rich = snmf._render_evidence("activation", items, 10)
        self.assertTrue(rich)
        self.assertIn("Token: `Rome`", text)
        self.assertIn("Context: `in Rome the`", text)
        self.assertIn("Score: `0.9`", text)

    def test_bare_strings_still_render_but_are_flagged_degraded(self):
        text, rich = snmf._render_evidence("activation", ["Rome", "Caesar"], 10)
        self.assertFalse(rich, "callers must be able to warn about this")
        self.assertIn("Token: `Rome`", text)

    def test_projection_source_renders_scores(self):
        items = [{"token": "Rome", "score": 12.5}]
        text, rich = snmf._render_evidence("projection", items, 10)
        self.assertTrue(rich)
        self.assertIn("Score: `12.5`", text)

    def test_results_section_is_extracted(self):
        reply = "Analysis:\nSome reasoning here.\n\nResults:\nRoman history."
        self.assertEqual(snmf._extract_results(reply), "Roman history.")

    def test_reply_without_the_format_is_used_whole(self):
        self.assertEqual(snmf._extract_results("  Roman history.  "),
                         "Roman history.")


class TrashShortCircuits(unittest.TestCase):
    """TRASH means 'these tokens share no concept'. Passing it to STAGE2 asks
    whether the string TRASH describes the concept, which is not a question
    about the feature."""

    class _Client:
        def __init__(self, stage1):
            self.stage1, self.classify_calls = stage1, 0

        def describe(self, _prompt):
            return self.stage1

        def classify(self, _prompt):
            self.classify_calls += 1
            return '{"is_member": true, "confidence": 0.99}'

    def test_trash_is_rejected_without_a_stage2_call(self):
        client = self._Client("Analysis:\nnothing.\n\nResults:\nTRASH")
        got = snmf._judge_evidence(client, "Ancient Rome", ["x"], 10, 0.85)
        self.assertTrue(got["trash"])
        self.assertFalse(got["accepted"])
        self.assertEqual(client.classify_calls, 0)

    def test_a_real_description_reaches_stage2(self):
        client = self._Client("Analysis:\nok.\n\nResults:\nAncient Roman rule.")
        got = snmf._judge_evidence(client, "Ancient Rome", ["x"], 10, 0.85)
        self.assertFalse(got["trash"])
        self.assertTrue(got["accepted"])
        self.assertEqual(client.classify_calls, 1)


class RhoScaleControl(unittest.TestCase):
    """rho is a ratio of mean coefficients, so a global magnitude gap between
    the two sentence sets lifts every feature equally. That is what layer 14's
    median of 3.05 and 63/100 over tau look like."""

    def setUp(self):
        torch.manual_seed(0)
        self.is_c = torch.zeros(400, dtype=torch.bool)
        self.is_c[:200] = True
        self.Y = torch.rand(50, 400) + 0.1

    def test_a_pure_scale_gap_inflates_plain_rho(self):
        Y = self.Y.clone()
        Y[:, self.is_c] *= 3.0
        rho = snmf.mass_ratio(Y, self.is_c)
        self.assertAlmostEqual(float(rho.median()), 3.0, delta=0.15)
        self.assertEqual(int((rho > 2.0).sum()), 50, "all of them clear tau")

    def test_the_normalized_ratio_removes_it(self):
        Y = self.Y.clone()
        Y[:, self.is_c] *= 3.0
        rho_n = snmf.mass_ratio_normalized(Y, self.is_c)
        self.assertAlmostEqual(float(rho_n.median()), 1.0, delta=0.1)

    def test_a_genuinely_selective_feature_survives_normalization(self):
        Y = self.Y.clone()
        Y[7, self.is_c] *= 6.0                     # one feature, not the side
        rho = snmf.mass_ratio(Y, self.is_c)
        rho_n = snmf.mass_ratio_normalized(Y, self.is_c)
        self.assertGreater(float(rho[7]), 3.0)
        self.assertGreater(float(rho_n[7]), 3.0)
        self.assertLess(float(rho_n.median()), 1.2)

    def test_summary_reports_the_inflation_factor(self):
        Y = self.Y.clone()
        Y[:, self.is_c] *= 3.0
        got = snmf.rho_summary(snmf.mass_ratio(Y, self.is_c), 2.0,
                               snmf.mass_ratio_normalized(Y, self.is_c))
        self.assertIn("normalized", got)
        self.assertAlmostEqual(got["scale_inflation"], 3.0, delta=0.3)


class RelativeConvergence(unittest.TestCase):
    """tol=1e-4 against a reconstruction error of 1e12 means 'any decrease at
    all', so patience never fires and layers stop at whatever cap they hit."""

    def test_absolute_tolerance_is_below_float32_resolution_at_1e12(self):
        err = 1.38e12
        self.assertLess(1e-4, err * 2 ** -24,
                        "the old tol is smaller than one representable step")

    def test_fit_info_reports_how_the_layer_stopped(self):
        torch.manual_seed(0)
        A = torch.rand(24, 60)
        _, _, info = snmf.semi_nmf(A, 4, max_iter=40, patience=5,
                                   device="cpu", verbose=False)
        self.assertIn(info["stop_reason"], ("patience", "max_iter"))
        self.assertEqual(info["converged"], info["stop_reason"] == "patience")
        self.assertLessEqual(info["best_iter"], info["stopped_at"])

    def test_a_tight_rtol_does_not_run_forever(self):
        torch.manual_seed(0)
        A = torch.rand(24, 60)
        _, _, info = snmf.semi_nmf(A, 4, max_iter=500, patience=3, rtol=1e-2,
                                   device="cpu", verbose=False)
        self.assertEqual(info["stop_reason"], "patience")

    def test_factorization_still_reconstructs(self):
        torch.manual_seed(0)
        A = torch.rand(24, 60)
        Z, Y, _ = snmf.semi_nmf(A, 8, max_iter=200, patience=50,
                                device="cpu", verbose=False)
        self.assertEqual(Z.shape, (24, 8))
        self.assertEqual(Y.shape, (8, 60))
        self.assertTrue((Y >= 0).all(), "Y must stay non-negative")


class PublishedRangeCells(unittest.TestCase):
    """The reference has three layer-range cells per side, not one."""

    def test_all_three_cells_reproduce_the_gemma_indices(self):
        expect_in = [(0, 25), (0, 8), (0, 12)]
        expect_out = [(0, 8), (9, 17), (13, 25)]
        for cell in range(3):
            got_in, got_out = snmf.default_layer_ranges(26, cell)
            self.assertEqual(got_in, expect_in[cell], f"in, cell {cell}")
            self.assertEqual(got_out, expect_out[cell], f"out, cell {cell}")

    def test_cell_zero_is_unchanged_on_the_lment_1b(self):
        self.assertEqual(snmf.default_layer_ranges(18), ((0, 17), (0, 5)))
        self.assertEqual(snmf.default_layer_ranges(18, 0), ((0, 17), (0, 5)))

    def test_a_cell_exists_that_covers_the_judges_layers(self):
        # job 871388 selected features at layers 9 and 14; cell 0's output
        # range is [0,5] and the one-sided guard correctly refused
        _, out = snmf.default_layer_ranges(18, 2)
        lo, hi = out
        for layer in (9, 14):
            self.assertTrue(lo <= layer <= hi,
                            f"layer {layer} outside {out} in cell 2")

    def test_ranges_never_leave_the_model(self):
        for n in (6, 12, 18, 26, 32):
            for cell in range(3):
                (a, b), (c, d) = snmf.default_layer_ranges(n, cell)
                for v in (a, b, c, d):
                    self.assertTrue(0 <= v <= n - 1, (n, cell, v))
                self.assertLessEqual(a, b)
                self.assertLessEqual(c, d)

    def test_an_unknown_cell_is_refused(self):
        with self.assertRaises(ValueError):
            snmf.default_layer_ranges(18, 5)


class OutputSideIsMeasurable(unittest.TestCase):
    """feature_activation hooks the INPUT to down_proj, so it cannot see a
    down_proj edit at the same layer. feature_readout can -- but only if it is
    handed the direction from BEFORE the edit.

    The earlier version of this class re-implemented the projection inline and
    never called snmf.feature_readout, so it passed while the real function was
    returning normalised rounding noise at delta=1.
    """

    def setUp(self):
        self.model = tiny_olmo2()
        self.layer = 2
        d_mlp = self.model.config.intermediate_size
        torch.manual_seed(1)
        self.Z = torch.zeros(d_mlp, 2)
        self.Z[:10, 0] = torch.randn(10)      # a real feature
        # column 1 left all-zero: the empty-support branch

    def _readout(self, dirs=None):
        return snmf.feature_readout(self.model, self.layer, self.Z, [0, 1],
                                    dirs)

    def test_empty_support_reads_zero(self):
        self.assertEqual(float(self._readout()[1]), 0.0)

    def test_delta_one_collapses_the_readout(self):
        dirs = snmf.feature_directions(self.model, self.layer, self.Z, [0, 1])
        before = float(self._readout(dirs)[0])
        snmf.ablate_layer(self.model, self.layer, self.Z, [0],
                          delta_in=0.0, delta_out=1.0, gamma=1.0)
        after = float(self._readout(dirs)[0])
        self.assertGreater(before, 0.0)
        self.assertLess(after, before * 1e-4,
                        "delta=1 is an exact projection; it must read as one")

    def test_recomputing_the_direction_reports_noise_instead(self):
        """Pin the bug itself, so the fix cannot be silently reverted."""
        before = float(self._readout()[0])
        snmf.ablate_layer(self.model, self.layer, self.Z, [0],
                          delta_in=0.0, delta_out=1.0, gamma=1.0)
        after_recomputed = float(self._readout()[0])
        # the edit really did land ...
        W_out = snmf.mlp_of(
            snmf.get_layers(self.model)[self.layer]).down_proj.weight.data.T
        resid = (W_out.T @ self.Z[:, 0].to(W_out.dtype)).norm()
        self.assertLess(float(resid), 1e-5, "the projection did not land")
        # ... yet the recomputed-direction number is nowhere near zero
        self.assertGreater(after_recomputed, before * 1e-2,
                           "if this ever fails the clamp moved; revisit "
                           "feature_readout's docstring")

    def test_directions_are_unaffected_by_a_later_edit(self):
        dirs = snmf.feature_directions(self.model, self.layer, self.Z, [0])
        saved = dirs[0].clone()
        snmf.ablate_layer(self.model, self.layer, self.Z, [0],
                          delta_in=0.0, delta_out=1.0, gamma=1.0)
        self.assertTrue(torch.equal(dirs[0], saved),
                        "directions must not be a view into the weights")

    def test_delta_two_leaves_the_magnitude_alone(self):
        # (I - 2P) flips the sign of the component; .abs() should see no drop
        dirs = snmf.feature_directions(self.model, self.layer, self.Z, [0])
        before = float(self._readout(dirs)[0])
        snmf.ablate_layer(self.model, self.layer, self.Z, [0],
                          delta_in=0.0, delta_out=2.0, gamma=1.0)
        after = float(self._readout(dirs)[0])
        self.assertAlmostEqual(after, before, delta=before * 1e-3)


if __name__ == "__main__":
    unittest.main()
