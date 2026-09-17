from dialogue_bench.__main__ import DEFAULT_MODELS, build_arg_parser


def test_cli_defaults_to_three_models():
    args = build_arg_parser().parse_args([])
    assert args.models is None
    assert len(DEFAULT_MODELS) == 3


def test_cli_accepts_repeatable_model_selection():
    args = build_arg_parser().parse_args(["--model", "one", "--model", "two"])
    assert args.models == ["one", "two"]


def test_cli_selects_openai_compatible_backend():
    assert build_arg_parser().parse_args([]).backend == "ollama"
    args = build_arg_parser().parse_args(["--backend", "openai", "--base-url", "http://mlx:8080"])
    assert (args.backend, args.base_url) == ("openai", "http://mlx:8080")
