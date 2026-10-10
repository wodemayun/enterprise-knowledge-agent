from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compose_declares_required_services_and_healthchecks():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "api:" in compose
    assert "qdrant:" in compose
    assert "postgres:" in compose
    assert compose.count("healthcheck:") >= 3


def test_example_environment_contains_no_real_secret():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=" in text
    assert "sk-" not in text


def test_sample_handbook_contains_demo_policies():
    handbook = (ROOT / "sample_data" / "employee_handbook.md").read_text(encoding="utf-8")

    assert "年假" in handbook
    assert "报销" in handbook
    assert "信息安全" in handbook
