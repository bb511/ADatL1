from scripts import generation, preflight


def test_preflight_composes_every_generated_core_sweep() -> None:
    specs = generation.build_paper_experiments(
        n_trials=1,
        seeds=(123,),
        include_cvar10=False,
    )

    errors = preflight.compose_experiment_matrix(
        specs.values(),
        launcher=generation.Launcher.SUBMITIT_SLURM_CLARIDEN,
    )

    assert len(specs) == 76
    assert errors == []


def test_preflight_generated_shells_pass_bash_parser() -> None:
    specs = generation.build_paper_experiments(
        n_trials=1,
        seeds=(123,),
        include_cvar10=False,
    )

    errors = preflight.validate_generated_shells(
        specs.values(),
        launcher=generation.Launcher.SUBMITIT_SLURM_CLARIDEN,
    )

    assert errors == []
