# Archived tests

These are the suites that existed before the client/node split, the YAML
configs and the run-folder design. They are kept for reference only — **they
are not run and will not pass**, because they assert against APIs that no
longer exist (`blacklist_path=`, `.npy` blacklists, `trainer.save_folder`
lookups, JSON configs, `run_smoke_*.sh`).

Every behaviour they checked was re-implemented from scratch in the current
suites. The mapping, one row per old assertion:

## test_exclusion.py

| Old assertion | Now covered by |
|---|---|
| masked rows are entirely ignore_index | `test_local_integration.py::test_get_labels_honours_instance_mask` |
| loss normaliser excludes masked tokens | same test (asserts the live-token count) |
| empty blacklist leaves labels bit-identical | `test_local_units.py::test_empty_blacklist_is_a_no_op` |
| instance_mask survives split_batch | `test_local_integration.py::test_split_batch_preserves_mask_and_index` |
| fully-masked batch is NaN without the guard | `test_local_units.py::test_fully_masked_batch_is_why_the_guard_exists` |
| blacklist `.npy` round-trips deduped/sorted/int64 | **obsolete** — `.npy` blacklists were removed; the equivalent is now `test_local_units.py::test_fetch_chunk_ids_dedupes_and_sorts` and `test_artifact_records_every_input_that_produced_it` |

The old file also vendored a copy of `get_labels`/`split_batch` so it could run
without olmo_core. The new suites import the real functions and *skip* when
olmo_core is unavailable — a vendored copy can drift from upstream and silently
stop testing the thing it claims to test.

## test_untaught_units.py

| Old assertion | Now covered by |
|---|---|
| control run: no mask, no metrics | `units::test_control_run_leaves_the_batch_untouched` |
| blacklisted rows masked, metric == 2, index untouched | `units::test_blacklisted_rows_are_masked_and_counted` |
| existing instance_mask is ANDed | `units::test_existing_mask_is_anded_not_clobbered` |
| guard keeps one row and records the leak | `units::test_all_masked_guard_keeps_one_row_and_records_the_leak` |
| guard_all_masked=False masks everything | `units::test_guard_can_be_disabled` |
| strict=True raises without 'index' | `units::test_strict_mode_raises_when_the_batch_has_no_index` |
| strict=False warns and continues | `units::test_non_strict_mode_warns_once_and_continues` |
| missing artifact fails at pre_train | `units::test_missing_artifact_fails_fast_with_instructions` |
| blacklist file parses objects and bare QIDs | `units::test_blacklist_file_parsing` |
| thresholds merge per key | `units::test_thresholds_merge_per_key_over_the_defaults` |
| bad thresholds rejected | `units::test_bad_thresholds_are_rejected` |
| case_sensitive selects cs/ci index | `units::test_case_sensitive_selects_the_index` |
| ES query shape and OR semantics | `units::test_entity_query_shape` |
| artifact records ids/index/thresholds/counts | `units::test_artifact_records_every_input_that_produced_it` |
| artifact is readable JSON | same test |
| artifact loads as a {chunk_id: qid} dict | same test |
| the dict masks the blacklisted rows | `units::test_the_node_half_never_needs_elasticsearch` |
| empty blacklist file is a no-op | `units::test_empty_blacklist_is_a_no_op` |
| expand_env / assert_paths_resolved | `units::test_env_var_expansion_and_unresolved_detection` |
| load_config sources env.sh; explicit value wins | `units::test_load_config_falls_back_to_env_sh` |
| YAML parses, comments and quoted '#' | `units::test_yaml_config_parsing` |
| flat job/train map onto upstream's schema | `units::test_to_upstream_produces_every_group_build_config_reads` |
| every group build_config reads is produced | same test + `integration::test_build_config_accepts_our_schema` |
| shipped configs: 3 groups, no _comment fields, train identical | `refactoring::test_configs_have_no_pseudo_comment_fields`, `refactoring::test_the_pair_differs_only_where_intended` |
| config copied verbatim | `units::test_prepare_control_run_folder` |
| control artifact explicitly empty | same test + `units::test_empty_artifact_is_explicit` |
| job.slurm exact format, no env vars | `units::test_generated_job_slurm_matches_the_required_format` |
| run_wrapper runs node env + torchrun on the copy | `units::test_generated_run_wrapper` |
| checkpoints/ created | `units::test_prepare_control_run_folder` |
| ablated folder carries the exclusion | `units::test_prepare_ablated_run_resolves_the_exclusion` |
| job.gpus flows to #SBATCH and torchrun | `units::test_generated_run_wrapper` + `refactoring::test_job_parameters_live_only_in_configs` |

Nothing was dropped except the `.npy` round-trip, whose subject no longer
exists in the design.
