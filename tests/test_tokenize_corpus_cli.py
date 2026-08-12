from scripts import tokenize_corpus


def test_profile_selects_its_default_config():
    pilot = tokenize_corpus.parse_args([])
    full = tokenize_corpus.parse_args(["--profile", "full"])

    assert pilot.profile == "pilot"
    assert pilot.config is None
    assert tokenize_corpus.DEFAULT_CONFIGS[pilot.profile] == (
        "configs/tokenized_data.yaml"
    )
    assert full.profile == "full"
    assert full.config is None
    assert tokenize_corpus.DEFAULT_CONFIGS[full.profile] == (
        "configs/tokenized_data_full.yaml"
    )


def test_explicit_config_override_is_preserved():
    args = tokenize_corpus.parse_args(
        ["--profile", "full", "--config", "configs/custom-full.yaml"]
    )
    assert args.config == "configs/custom-full.yaml"
