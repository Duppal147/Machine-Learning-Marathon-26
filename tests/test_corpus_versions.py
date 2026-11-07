"""The corpus must use the arXiv versions the competition's answers were read from."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_download_keeps_the_pinned_version():
    download = load("download_papers")
    assert download.pinned_pdf_url("https://arxiv.org/pdf/2412.06288v1", "latest") == \
        "https://arxiv.org/pdf/2412.06288v1"
    assert download.pinned_pdf_url("https://arxiv.org/abs/2505.09598v4", "latest") == \
        "https://arxiv.org/pdf/2505.09598v4"


def test_download_without_a_version_uses_the_fallback():
    download = load("download_papers")
    assert download.pinned_pdf_url("https://arxiv.org/pdf/2412.06288", "latest") == "latest"
    assert download.pinned_pdf_url("https://example.com/report.pdf", "latest") == "latest"


def test_version_check_reads_the_stamp():
    check = load("check_corpus_versions")
    url = "https://arxiv.org/pdf/2412.06288v1"
    assert check.compare(url, "arXiv:2412.06288v1 [cs.CY]") == "match"
    assert check.compare(url, "arXiv:2412.06288v4 [cs.CY]") == "mismatch v4"
    assert check.compare(url, "no stamp") == "no-stamp"
    assert check.compare("https://arxiv.org/pdf/2412.06288", "") == "unpinned"
