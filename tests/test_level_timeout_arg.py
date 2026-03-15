from main import parse_args


def test_parse_args_level_timeout_sec_override(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--levels", "1", "--level-timeout-sec", "25"],
    )
    args = parse_args()
    assert args.levels == 1
    assert args.level_timeout_sec == 25.0
