import json
import stat

from scripts import generation


def test_default_paper_registry_fits_current_experiment_matrix() -> None:
    specs = generation.build_paper_experiments(include_cvar10=False)

    assert hasattr(generation, "ExperimentSpecification")
    assert not hasattr(generation, "StudySpec")
    assert len(specs) == 76
    assert "physics_dsae_cap" in specs
    assert "cifar10_dsae_cap" not in specs
    assert specs["physics_ae_cap"].experiment == "physics/ae_agnostic"
    assert specs["physics_ae_semi_cvar25"].experiment == "physics/ae"
    assert specs["cchamber_ae_cap_metadata_nearest"].experiment == "cchamber/ae_agnostic"
    assert "cchamber_ae_semi_cvar25" not in specs


def test_sweep_manifest_separates_tuned_params_from_fixed_factors() -> None:
    spec = generation.make_experiment_specification(
        dataset=generation.Dataset.PHYSICS,
        model=generation.Model.AE,
        strategy=generation.Strategy.CAP,
        n_trials=7,
        seeds=(123,),
    )
    commands = generation.sweep_commands_for(
        spec,
        launcher=generation.Launcher.NONE,
        trainer="gpu",
        devices="[0]",
        cpus_per_task=1,
        gpus_per_node=1,
        timeout_min=None,
    )

    manifest = generation.build_manifest(spec, {"sweep": commands})

    assert "algorithm.optimizer.lr" in manifest["tuned_params"]
    assert "data.batch_size=16384" in manifest["fixed_overrides"]
    assert (
        "optimized_metric_config.main_metric.callback.name=cap" in manifest["strategy_overrides"]
    )
    assert "hydra.sweeper.study_name=cap_vs_mse" in manifest["sweeper_overrides"]
    assert manifest["factors"]["reported_over"] == ["signal_dataset", "seed"]
    assert manifest["experiment"]["objective_name"] == "cap"

    command = commands[0]
    assert "hparams_search=ae_optuna" in command
    assert "hydra.sweeper.n_trials=7" in command
    assert "optimized_metric_config.main_metric.callback.name=cap" in command
    assert 'paths.raw_data_dir="${RAW_DATA_DIR}"' in command
    assert "'~evaluation.evaluator.ckpts.summary.operational_drift_ema'" in command


def test_generation_registry_includes_paired_causal_chamber() -> None:
    specs = generation.build_paper_experiments(
        n_trials=1,
        seeds=(123,),
        include_cvar10=True,
    )

    spec = specs["cchamber_ae_cap_metadata_nearest"]
    assert spec.dataset == generation.Dataset.CCHAMBER
    assert spec.strategy == generation.Strategy.CAP_METADATA_NEAREST
    assert spec.experiment == "cchamber/ae_agnostic"
    assert spec.hparams_search == "ae_optuna"
    assert "data.pairing_strategy=metadata_nearest" in spec.strategy_overrides

    encoder_spec = specs["cchamber_ae_cap_encoder_nearest"]
    assert "data.pairing_strategy=random" in encoder_spec.strategy_overrides
    assert "callbacks.cap_ref.pairing_type=precomputed" in encoder_spec.strategy_overrides
    assert (
        "callbacks.cap_ref.pairing_index_path=$CCHAMBER_VALID_PAIR_TABLE"
        in encoder_spec.strategy_overrides
    )
    assert (
        "evaluation.callbacks.cap_ref.pairing_index_path=$CCHAMBER_TEST_PAIR_TABLE"
        in encoder_spec.strategy_overrides
    )

    cchamber_strategies = {
        item.strategy for item in specs.values() if item.dataset == generation.Dataset.CCHAMBER
    }
    assert cchamber_strategies == {
        generation.Strategy.CAP_METADATA_NEAREST,
        generation.Strategy.CAP_ENCODER_NEAREST,
        generation.Strategy.CAP_RANDOM,
        generation.Strategy.DRIFT,
        generation.Strategy.WASSERSTEIN,
    }


def test_generate_scripts_writes_executable_script_and_manifest(tmp_path) -> None:
    spec = generation.make_experiment_specification(
        dataset=generation.Dataset.CIFAR10,
        model=generation.Model.VAE,
        strategy=generation.Strategy.SEMI_CVAR25,
        n_trials=2,
        seeds=(123, 456),
    )

    written = generation.generate_scripts(
        [spec],
        output_dir=tmp_path,
        stage=generation.Stage.SWEEP,
        launcher=generation.Launcher.NONE,
        trainer="cpu",
        devices="1",
    )

    spec_dir = tmp_path / spec.name
    script_path = spec_dir / "sweep.sh"
    manifest_path = spec_dir / "manifest.json"

    assert script_path in written
    assert manifest_path in written
    assert script_path.stat().st_mode & stat.S_IXUSR

    script = script_path.read_text(encoding="utf-8")
    assert "trainer=cpu" in script
    assert "seed=123" in script
    assert "seed=456" in script
    assert "evaluation.callbacks.reco" not in script

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment"]["name"] == "cifar10_vae_semi_cvar25"
    assert manifest["commands"]["sweep"]
    assert "algorithm.optimizer.lr" in manifest["tuned_params"]


def test_encoder_pairing_script_requires_pair_tables(tmp_path) -> None:
    spec = generation.make_experiment_specification(
        dataset=generation.Dataset.CCHAMBER,
        model=generation.Model.AE,
        strategy=generation.Strategy.CAP_ENCODER_NEAREST,
        n_trials=2,
        seeds=(123,),
    )

    generation.generate_scripts(
        [spec],
        output_dir=tmp_path,
        stage=generation.Stage.SWEEP,
        launcher=generation.Launcher.NONE,
        trainer="cpu",
        devices="1",
    )

    script = (tmp_path / spec.name / "sweep.sh").read_text(encoding="utf-8")
    assert "CCHAMBER_VALID_PAIR_TABLE" in script
    assert "CCHAMBER_TEST_PAIR_TABLE" in script
    assert "callbacks.cap_ref.pairing_type=precomputed" in script
    assert "callbacks.cap_ref.pairing_index_path=$CCHAMBER_VALID_PAIR_TABLE" in script
    assert "evaluation.callbacks.cap_ref.pairing_index_path=$CCHAMBER_TEST_PAIR_TABLE" in script


def test_display_path_handles_paths_outside_repo(tmp_path) -> None:
    outside_repo = tmp_path / "generated" / "sweep.sh"

    assert generation.display_path(outside_repo) == str(outside_repo)
