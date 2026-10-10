from app.core.config import Settings


def test_settings_use_offline_compatible_local_defaults():
    settings = Settings(_env_file=None)

    assert settings.app_name == "Enterprise Knowledge Agent"
    assert settings.llm_provider == "deterministic"
    assert settings.evidence_threshold == 0.3
    assert settings.retrieval_limit == 4
