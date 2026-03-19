from dataclasses import replace
from pathlib import Path

import pytest

from quant_etf.config import ConfigValidationError, load_app_config, validate_app_config
from quant_etf.config.schema import AppConfig


def test_load_app_config_reads_template_files() -> None:
    config = load_app_config("configs", env_prefix=None)

    assert isinstance(config, AppConfig)
    assert config.strategy.lookback_windows == (20, 60, 120)
    assert config.strategy.score_weights == (0.4, 0.4, 0.2)
    assert config.strategy.buy_top_n == 3
    assert config.strategy.hold_buffer_n == 6
    assert config.universe.min_listed_days == 120
    assert config.data.raw_dir == config.app.project_root / "data/raw"
    assert config.live.signal_output_path == config.app.project_root / "data/reports/live_signals.csv"


def test_validate_app_config_rejects_invalid_weights() -> None:
    config = load_app_config("configs", env_prefix=None)
    broken = replace(config, strategy=replace(config.strategy, score_weights=(0.5, 0.3, 0.1)))

    with pytest.raises(ConfigValidationError):
        validate_app_config(broken)


def test_validate_app_config_rejects_invalid_execution_settings() -> None:
    config = load_app_config("configs", env_prefix=None)

    with pytest.raises(ConfigValidationError):
        validate_app_config(replace(config, strategy=replace(config.strategy, execution_delay_days=0)))

    with pytest.raises(ConfigValidationError):
        validate_app_config(replace(config, strategy=replace(config.strategy, execution_timing="same_open")))
