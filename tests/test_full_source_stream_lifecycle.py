import pytest

import scripts.build_fineweb_edu_corpus as corpus_module


class CloseTrackingStream:
    def __init__(self, events):
        self.events = events
        self.close_calls = 0

    def close(self):
        self.events.append("close")
        self.close_calls += 1


def test_managed_source_stream_closes_after_success(monkeypatch):
    events = []
    source_stream = CloseTrackingStream(events)
    config = object()
    expected_manifest = {"status": "complete"}

    monkeypatch.setattr(
        corpus_module,
        "open_fineweb_edu_stream",
        lambda actual_config: source_stream,
    )

    def fake_build_corpus(records, actual_config):
        assert records is source_stream
        assert actual_config is config
        events.append("build")
        return expected_manifest

    monkeypatch.setattr(corpus_module, "build_corpus", fake_build_corpus)

    result = corpus_module._build_corpus_with_managed_stream(config)

    assert result is expected_manifest
    assert events == ["build", "close"]
    assert source_stream.close_calls == 1


def test_managed_source_stream_closes_after_build_failure(monkeypatch):
    events = []
    source_stream = CloseTrackingStream(events)
    config = object()

    monkeypatch.setattr(
        corpus_module,
        "open_fineweb_edu_stream",
        lambda actual_config: source_stream,
    )

    def failing_build_corpus(records, actual_config):
        assert records is source_stream
        assert actual_config is config
        events.append("build")
        raise RuntimeError("controlled build failure")

    monkeypatch.setattr(corpus_module, "build_corpus", failing_build_corpus)

    with pytest.raises(RuntimeError, match="controlled build failure"):
        corpus_module._build_corpus_with_managed_stream(config)

    assert events == ["build", "close"]
    assert source_stream.close_calls == 1


def test_managed_source_stream_accepts_non_closeable_source(monkeypatch):
    source_stream = object()
    config = object()
    expected_manifest = {"status": "complete"}

    monkeypatch.setattr(
        corpus_module,
        "open_fineweb_edu_stream",
        lambda actual_config: source_stream,
    )
    monkeypatch.setattr(
        corpus_module,
        "build_corpus",
        lambda records, actual_config: expected_manifest,
    )

    assert (
        corpus_module._build_corpus_with_managed_stream(config)
        is expected_manifest
    )
