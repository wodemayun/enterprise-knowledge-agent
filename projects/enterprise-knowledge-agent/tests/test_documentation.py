from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_complete_runbook():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in [
        "pip install",
        "pytest",
        "uvicorn",
        "docker compose",
        "/api/v1/documents",
        "/api/v1/questions",
        "故障排查",
    ]:
        assert required in readme


def test_learning_guide_connects_csharp_and_python():
    guide = (ROOT / "docs" / "learning-guide.md").read_text(encoding="utf-8")
    assert "C#" in guide
    assert "Protocol" in guide
    assert "pytest" in guide
    assert guide.count("练习") >= 5


def test_interview_guide_is_honest_and_actionable():
    guide = (ROOT / "docs" / "interview-guide.md").read_text(encoding="utf-8")
    assert "60 秒" in guide
    assert "个人项目" in guide
    assert "没有实现" in guide
    assert "追问" in guide


def test_architecture_contains_both_mermaid_flows():
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert architecture.count("```mermaid") >= 2
    assert "create_ticket" in architecture
    assert "Qdrant" in architecture
