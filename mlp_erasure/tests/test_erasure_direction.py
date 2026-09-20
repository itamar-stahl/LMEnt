"""Tests for the bug that made an "erased" model score BETTER on its concept.

The failure: `snmf.py erase` defaulted to --delta-in 4 --delta-out 4, and
ablate_layer applies (I - delta*P) onto a UNIT direction, so the targeted
component scales by |1 - delta| = 3. The saved checkpoint carried the concept
three times as strongly as its control, and nothing downstream could tell that
apart from an erasure that simply failed.

These tests pin four things:

  1. the direction of the edit, measured on the residual stream rather than on
     a weight statistic -- delta 1 removes, delta 4 amplifies;
  2. the guard that now refuses to save an amplifying edit;
  3. the CLI defaults, which are what actually got run;
  4. that erase and verify accept the same deltas, so you cannot save a cell
     you were not allowed to verify.

Plus the OLMo-2 parameter-index bug in the fork's RMU method.

    python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v
"""
import argparse
import sys
import unittest
from pathlib import Path

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
FORK = ROOT / "Ember-on-LMEnt"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FORK))
# ember.utils imports `factorization.seminmf` from the vendored SNMF reference,
# and ember/erasure/methods/__init__ pulls ember.utils in transitively.
sys.path.insert(0, str(FORK / "external" / "snmf"))
sys.path.insert(0, str(FORK / "external" / "wmdp"))

import snmf   # noqa: E402


def tiny_olmo2(n_layers=6, hidden=32, mlp=88):
    torch.manual_seed(0)
    return Olmo2ForCausalLM(Olmo2Config(
        vocab_size=64, hidden_size=hidden, intermediate_size=mlp,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=32,
        tie_word_embeddings=False))


def sparse_feature(model, n_support=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    Z = torch.zeros(model.config.intermediate_size, 1)
    Z[:n_support, 0] = torch.randn(n_support, generator=g)
    return Z


class ErasureScaleArithmetic(unittest.TestCase):
    """|1 - delta|, not delta. The whole bug in one function."""

    def test_delta_one_is_the_only_exact_removal(self):
        self.assertEqual(snmf.erasure_scale(1.0), 0.0)

    def test_published_sweep_values_do_not_shrink(self):
        # configs/snmf_gemma.yaml: in_deltas/out_deltas = [1.0, 4.0, 7.0, 10.0]
        self.assertEqual(snmf.erasure_scale(4.0), 3.0)
        self.assertEqual(snmf.erasure_scale(7.0), 6.0)
        self.assertEqual(snmf.erasure_scale(10.0), 9.0)

    def test_two_is_the_boundary(self):
        self.assertEqual(snmf.erasure_scale(2.0), 1.0)   # sign flip only
        self.assertLess(snmf.erasure_scale(1.9), 1.0)    # still shrinks
        self.assertGreater(snmf.erasure_scale(2.1), 1.0)  # already amplifies


class ResidualStreamDirection(unittest.TestCase):
    """Measured on what the rest of the network sees, not on the weights.

    feature_readout and the activation table both look at one side of the edit.
    This hooks the MLP's output and asks how much of the feature direction the
    layer writes into the residual stream -- the quantity that decides whether
    a downstream MC eval gets better or worse.
    """

    LAYER = 1

    def _contribution_ratio(self, delta):
        model = tiny_olmo2()
        Z = sparse_feature(model)
        tokens = torch.randint(0, 64, (4, 16),
                               generator=torch.Generator().manual_seed(1))
        f_out = snmf.feature_directions(model, self.LAYER, Z, [0])[0]

        def contribution():
            grabbed = []
            h = snmf.mlp_of(snmf.get_layers(model)[self.LAYER]
                            ).register_forward_hook(
                lambda m, i, o: grabbed.append(o.detach()))
            try:
                with torch.no_grad():
                    model(input_ids=tokens)
            finally:
                h.remove()
            return (grabbed[0] @ f_out).abs().mean().item()

        before = contribution()
        snmf.ablate_layer(model, self.LAYER, Z, [0], delta, delta, gamma=0.95)
        return contribution() / before

    def test_delta_one_removes_concept_from_the_residual_stream(self):
        self.assertLess(self._contribution_ratio(1.0), 0.5)

    def test_the_old_default_amplified_it(self):
        # This is the regression. delta 4 was the shipped default for both
        # `erase` and `verify`, and it made the concept STRONGER.
        self.assertGreater(self._contribution_ratio(4.0), 1.5)

    def test_amplification_grows_with_delta(self):
        ratios = [self._contribution_ratio(d) for d in (1.0, 4.0, 7.0, 10.0)]
        for lo, hi in zip(ratios, ratios[1:]):
            self.assertLess(lo, hi)
        self.assertLess(ratios[0], 1.0)      # delta 1 shrinks
        self.assertGreater(ratios[1], 1.0)   # delta 4 already does not


class AmplificationGuard(unittest.TestCase):
    def test_the_old_default_is_now_refused(self):
        with self.assertRaises(SystemExit) as cm:
            snmf.check_delta_erases(4.0, 4.0, False, "erase")
        msg = str(cm.exception)
        self.assertIn("not an erasure", msg)
        self.assertIn("3.0", msg)            # the factor, stated
        self.assertIn("--allow-amplification", msg)

    def test_delta_one_passes(self):
        snmf.check_delta_erases(1.0, 1.0, False, "erase")

    def test_anything_below_two_passes(self):
        for d in (0.25, 0.5, 1.0, 1.5, 1.99):
            snmf.check_delta_erases(d, d, False, "erase")

    def test_sign_flip_at_exactly_two_is_refused(self):
        # magnitude untouched, so it erases nothing
        with self.assertRaises(SystemExit):
            snmf.check_delta_erases(2.0, 2.0, False, "erase")

    def test_one_amplifying_side_is_enough_to_refuse(self):
        with self.assertRaises(SystemExit) as cm:
            snmf.check_delta_erases(1.0, 7.0, False, "erase")
        self.assertIn("--delta-out", str(cm.exception))
        self.assertIn("down_proj", str(cm.exception))

    def test_zero_side_is_a_deliberate_one_sided_edit(self):
        # check_both_sides_applied already governs this case; the delta guard
        # must not second-guess it.
        snmf.check_delta_erases(1.0, 0.0, False, "erase")

    def test_explicit_sweep_flag_is_honoured(self):
        snmf.check_delta_erases(10.0, 10.0, True, "erase")


def _subparser_defaults(command):
    """The default delta_in/delta_out argparse would hand the command body.

    Built by running snmf.main() with a stub subcommand body, so the test reads
    the real parser rather than a copy of it that could drift.
    """
    captured = {}
    original = getattr(snmf, f"cmd_{command}")
    setattr(snmf, f"cmd_{command}", lambda a: captured.update(vars(a)))
    argv_backup = sys.argv
    # every required flag gets a dummy; the body is stubbed so none are read
    required = {
        "erase": ["--model", "m", "--out", "o"],
        "verify": ["--model", "m", "--out", "o", "--concept", "c",
                   "--concept-sentences", "cs", "--neutral-sentences", "ns"],
    }[command]
    try:
        sys.argv = ["snmf.py", command] + required
        snmf.main()
    finally:
        sys.argv = argv_backup
        setattr(snmf, f"cmd_{command}", original)
    return captured


class CLIDefaults(unittest.TestCase):
    """What actually got run. The defaults are the bug, not a footnote.

    `snmf.py erase` shipped with --delta-in 4 --delta-out 4, so the documented
    invocation in MLP_ERASURE.md produced an amplifying edit without anyone
    passing a delta at all.
    """

    def test_erase_defaults_to_exact_removal(self):
        got = _subparser_defaults("erase")
        self.assertEqual(got["delta_in"], 1.0)
        self.assertEqual(got["delta_out"], 1.0)

    def test_verify_defaults_match_erase(self):
        # verify must measure what erase would write
        erase = _subparser_defaults("erase")
        verify = _subparser_defaults("verify")
        self.assertEqual((erase["delta_in"], erase["delta_out"]),
                         (verify["delta_in"], verify["delta_out"]))

    def test_defaults_pass_their_own_guard(self):
        for command in ("erase", "verify"):
            got = _subparser_defaults(command)
            snmf.check_delta_erases(got["delta_in"], got["delta_out"],
                                    got["allow_amplification"], command)

    def test_amplification_is_off_by_default(self):
        for command in ("erase", "verify"):
            self.assertFalse(_subparser_defaults(command)["allow_amplification"])

    def test_the_old_defaults_would_now_fail_the_guard(self):
        # the exact configuration that produced the bad checkpoint
        with self.assertRaises(SystemExit):
            snmf.check_delta_erases(4.0, 4.0, False, "erase")


class AppliedSideCounting(unittest.TestCase):
    """A layer that edited nothing must not count towards either side.

    ablate_layer returns 0 when every selected feature lost its support to the
    coverage mask. cmd_erase used to increment applied_in/applied_out from the
    delta alone, so check_both_sides_applied saw a side as covered by a layer
    that had in fact been skipped.
    """

    def test_ablate_layer_reports_zero_when_support_is_empty(self):
        model = tiny_olmo2()
        Z = torch.zeros(model.config.intermediate_size, 1)   # all-zero feature
        n, kept = snmf.ablate_layer(model, 1, Z, [0], 1.0, 1.0)
        self.assertEqual(n, 0)

    def test_empty_support_leaves_the_weights_untouched(self):
        model = tiny_olmo2()
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        Z = torch.zeros(model.config.intermediate_size, 1)
        snmf.ablate_layer(model, 1, Z, [0], 1.0, 1.0)
        for k, v in model.named_parameters():
            self.assertTrue(torch.equal(v.detach(), before[k]), k)


def _vendored_get_params():
    """WMDP's real get_params, loaded from source.

    external/wmdp/rmu/utils.py imports `datasets` at module scope for a corpus
    loader this project never calls, so a plain import needs a dependency that
    has nothing to do with parameter selection. Compiling just the function
    keeps the test bound to the vendored code -- if upstream changes how it
    indexes parameters, this test sees it.
    """
    import ast
    source = (FORK / "external" / "wmdp" / "rmu" / "utils.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_params":
            namespace = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         "<wmdp get_params>", "exec"), namespace)
            return namespace["get_params"]
    raise AssertionError("get_params vanished from external/wmdp/rmu/utils.py")


class ForkRMUParamIndex(unittest.TestCase):
    """ember/erasure/methods/rmu.py hardcoded WMDP's param_ids=[6].

    On OLMo-2 that is mlp.gate_proj, so RMU trained the wrong matrix and
    down_proj -- the matrix the method is defined to move -- never changed.
    Job 871246 measured down_proj_index 8 on the real 1B.
    """

    def setUp(self):
        try:
            from ember.erasure.methods import rmu as fork_rmu
        except Exception as exc:                      # pragma: no cover
            self.skipTest(f"fork rmu not importable: {exc}")
        self.fork_rmu = fork_rmu
        self.model = tiny_olmo2()

    def test_resolved_index_is_down_proj_not_wmdp_six(self):
        got = self.fork_rmu._down_proj_param_ids(self.model)
        names = [n for n, _ in self.model.model.layers[0].named_parameters()]
        self.assertTrue(names[got[0]].endswith("mlp.down_proj.weight"),
                        names[got[0]])
        self.assertNotEqual(got, self.fork_rmu.FIXED_PARAM_IDS)

    def test_it_matches_the_index_measured_on_the_real_1b(self):
        self.assertEqual(self.fork_rmu._down_proj_param_ids(self.model), [8])

    def test_wmdp_get_params_now_selects_down_proj(self):
        get_params = _vendored_get_params()
        ids = self.fork_rmu._down_proj_param_ids(self.model)
        picked = get_params(self.model, [2], ids)
        self.assertEqual(len(picked), 1)
        self.assertIs(picked[0], self.model.model.layers[2].mlp.down_proj.weight)

    def test_the_old_hardcoded_index_picked_the_wrong_matrix(self):
        """The regression, stated positively: [6] is gate_proj on OLMo-2.

        RMU is defined to move down_proj. With WMDP's index it optimised
        gate_proj and left down_proj bit-identical, so the method's own
        intervention never happened.
        """
        get_params = _vendored_get_params()
        picked = get_params(self.model, [2], self.fork_rmu.FIXED_PARAM_IDS)
        self.assertIsNot(picked[0],
                         self.model.model.layers[2].mlp.down_proj.weight)
        self.assertIs(picked[0], self.model.model.layers[2].mlp.gate_proj.weight)


if __name__ == "__main__":
    unittest.main()
