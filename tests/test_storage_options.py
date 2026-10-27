"""A location's own fsspec options: passed unchanged, secrets only as `{"env": NAME}`, nothing shown but names."""

from __future__ import annotations

import json
from pathlib import Path

import fsspec.core
import pytest

from tests.test_delivery import _to, run_root  # noqa: F401
from tests.test_workflow_runner import _resolve, _stored
from womblex.cloud import delivery
from womblex.cloud.workflow_runner import run_workflow_local
from womblex.cloud.workflow_store import workflow_digest
from womblex.store.remote import RemoteStore, location_options_from_env, storage_options_from_env
from womblex.store.storage_options import (
    describe_options,
    merge_options,
    parse_options,
    resolve_options,
    secrets_outside_env,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in ("WOMBLEX_STORE_URI", "WOMBLEX_INGEST_URI", "WOMBLEX_STORE_STORAGE_OPTIONS",
                 "WOMBLEX_INGEST_STORAGE_OPTIONS", "WOMBLEX_S3_ENDPOINT", "AWS_ENDPOINT_URL",
                 "WOMBLEX_S3_ACCESS_KEY_ID", "WOMBLEX_S3_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
                 "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)


def test_options_must_be_a_json_object_and_the_error_names_where_they_came_from() -> None:
    assert parse_options('{"account_name": "acme"}', "$X") == {"account_name": "acme"}
    with pytest.raises(ValueError, match=r"\$X is not valid JSON"):
        parse_options("{nope", "$X")
    with pytest.raises(ValueError, match=r"\$X must be a JSON object of storage options, not list"):
        parse_options("[1]", "$X")


def test_an_env_reference_is_resolved_where_it_nests_and_an_unset_one_names_option_and_variable(monkeypatch) -> None:
    monkeypatch.setenv("ACME_KEY", "k-value")
    assert resolve_options({"account_name": "acme", "account_key": {"env": "ACME_KEY"},
                            "client_kwargs": {"token": {"env": "ACME_KEY"}}}) == {
        "account_name": "acme", "account_key": "k-value", "client_kwargs": {"token": "k-value"}}
    monkeypatch.delenv("ACME_KEY")
    with pytest.raises(ValueError, match=r"client_kwargs\.token reads environment variable ACME_KEY, which is not set"):
        resolve_options({"client_kwargs": {"token": {"env": "ACME_KEY"}}})
    monkeypatch.setenv("ACME_KEY", "")
    with pytest.raises(ValueError, match="not set"):
        resolve_options({"account_key": {"env": "ACME_KEY"}})


def test_a_secret_written_as_a_value_is_found_and_a_reference_or_plain_option_is_not() -> None:
    assert secrets_outside_env({"account_key": "abc", "client_kwargs": {"aws_secret_access_key": "x"},
                                "account_name": "acme", "endpoint_url": "https://x"}) == [
        "account_key", "client_kwargs.aws_secret_access_key"]
    assert secrets_outside_env({"account_key": {"env": "K"}, "sas_token": {"env": "T"}}) == []
    assert secrets_outside_env(None) == []


def test_what_is_shown_is_a_variable_name_or_a_mask_never_a_value() -> None:
    shown = describe_options({"account_name": "acme", "account_key": {"env": "ACME_KEY"}, "password": "hunter2",
                              "client_kwargs": {"endpoint_url": "https://acct.r2.cloudflarestorage.com"}})
    assert shown == {"account_name": "acme", "account_key": {"env": "ACME_KEY"}, "password": "••••••••",
                     "client_kwargs": {"endpoint_url": "https://acct.r2.cloudflarestorage.com"}}
    assert "hunter2" not in json.dumps(shown)


def test_merge_is_key_by_key_with_nested_dictionaries_merged_too() -> None:
    assert merge_options({"key": "a", "client_kwargs": {"endpoint_url": "e", "region_name": "r"}},
                         {"client_kwargs": {"region_name": "x"}, "anon": False}) == {
        "key": "a", "client_kwargs": {"endpoint_url": "e", "region_name": "x"}, "anon": False}


def test_explicit_options_win_over_the_s3_variables_which_win_over_the_aws_ones(monkeypatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    assert storage_options_from_env("s3://b/x")["key"] == "aws-id"
    monkeypatch.setenv("WOMBLEX_S3_ACCESS_KEY_ID", "store-id")
    monkeypatch.setenv("WOMBLEX_S3_SECRET_ACCESS_KEY", "store-secret")
    assert storage_options_from_env("s3://b/x")["key"] == "store-id"
    monkeypatch.setenv("R2_KEY", "r2-id")
    opts = storage_options_from_env("s3://b/x", explicit={"key": {"env": "R2_KEY"}})
    assert (opts["key"], opts["secret"]) == ("r2-id", "store-secret")  # an unnamed option keeps the lower layer


def test_an_s3_compatible_store_needs_configuration_only(monkeypatch) -> None:
    monkeypatch.setenv("WOMBLEX_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("R2_SECRET", "s")
    opts = storage_options_from_env("s3://b/x", explicit={
        "key": "id", "secret": {"env": "R2_SECRET"},
        "client_kwargs": {"endpoint_url": "https://acct.r2.cloudflarestorage.com"}})
    assert opts["client_kwargs"] == {"endpoint_url": "https://acct.r2.cloudflarestorage.com"}
    assert (opts["key"], opts["secret"]) == ("id", "s")


def test_another_backend_gets_exactly_the_options_given_and_none_of_the_s3_ones(monkeypatch) -> None:
    monkeypatch.setenv("WOMBLEX_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("ACME_KEY", "az-key")
    assert storage_options_from_env("az://container/x") == {}
    assert storage_options_from_env("az://container/x", explicit={
        "account_name": "acme", "account_key": {"env": "ACME_KEY"}}) == {
        "account_name": "acme", "account_key": "az-key"}


def test_the_environment_gives_each_location_its_own_options(monkeypatch) -> None:
    monkeypatch.setenv("WOMBLEX_STORE_URI", "s3://store-bucket/womblex")
    monkeypatch.setenv("WOMBLEX_INGEST_URI", "az://ingest-container/in")
    monkeypatch.setenv("WOMBLEX_STORE_STORAGE_OPTIONS", '{"client_kwargs": {"endpoint_url": "https://r2.example"}}')
    monkeypatch.setenv("WOMBLEX_INGEST_STORAGE_OPTIONS", '{"account_name": "acme"}')
    assert location_options_from_env("s3://store-bucket/womblex/runs/run-1") == {
        "client_kwargs": {"endpoint_url": "https://r2.example"}}
    assert location_options_from_env("az://ingest-container/in/2026") == {"account_name": "acme"}
    assert location_options_from_env("s3://store-bucket/womblex-other") is None  # a sibling, not inside
    assert location_options_from_env("s3://elsewhere/x") is None
    assert storage_options_from_env("az://ingest-container/in/x") == {"account_name": "acme"}


def test_an_environment_option_that_is_not_an_object_is_refused_naming_the_variable(monkeypatch) -> None:
    monkeypatch.setenv("WOMBLEX_STORE_URI", "s3://b")
    monkeypatch.setenv("WOMBLEX_STORE_STORAGE_OPTIONS", "just-a-string")
    with pytest.raises(ValueError, match=r"\$WOMBLEX_STORE_STORAGE_OPTIONS is not valid JSON"):
        storage_options_from_env("s3://b/x")


def test_the_options_reach_fsspec_unchanged(monkeypatch) -> None:
    seen: dict = {}
    real = fsspec.core.url_to_fs

    def spy(url, **kwargs):
        seen.update(kwargs)
        return real("memory://x")

    monkeypatch.setattr(fsspec.core, "url_to_fs", spy)
    monkeypatch.setenv("ACME_KEY", "az-key")
    RemoteStore.from_uri("az://container/x", explicit={"account_name": "acme", "account_key": {"env": "ACME_KEY"},
                                                       "anon": False})
    assert seen == {"account_name": "acme", "account_key": "az-key", "anon": False}


def test_a_destination_may_carry_options_but_not_a_secret_as_a_value(tmp_path: Path) -> None:
    _stored(["normalise"], destinations=[_to(tmp_path, storage_options={"anon": False})])
    _stored(["normalise"], destinations=[_to(tmp_path, storage_options={"account_key": {"env": "ACME_KEY"}})])
    with pytest.raises(ValueError, match=r'storage_options\.client_kwargs\.aws_secret_access_key holds a secret as a value; write it as \{"env": "NAME"\}'):
        _stored(["normalise"], destinations=[_to(tmp_path, storage_options={"client_kwargs": {"aws_secret_access_key": "abc"}})])


def test_a_destination_that_gives_no_options_keeps_the_digest_it_had(tmp_path: Path) -> None:
    import hashlib

    spec = _stored(["normalise"], destinations=[_to(tmp_path)]).spec
    payload = spec.model_dump(mode="json")
    for d in payload["destinations"]:
        del d["storage_options"]  # the dump as it was before the field existed
    del payload["collection"]
    del payload["inbound"]
    expected = "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert workflow_digest(spec) == expected
    with_options = _stored(["normalise"], destinations=[_to(tmp_path, storage_options={"anon": False})]).spec
    assert workflow_digest(with_options) != expected


def test_delivery_opens_the_destination_with_its_own_options(run_root: Path, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    opened: list[dict | None] = []
    real = RemoteStore.from_uri.__func__  # type: ignore[attr-defined]

    def spy(cls, uri, **kw):
        opened.append(kw.get("explicit"))
        return real(cls, uri, **{k: v for k, v in kw.items() if k != "explicit"})

    monkeypatch.setattr(RemoteStore, "from_uri", classmethod(spy))
    stored = _stored(["normalise"], destinations=[_to(tmp_path, storage_options={"anon": False})])
    result = run_workflow_local(stored, run_root / "documents", resolve_preset=_resolve)
    assert result.exit_code == 0 and {"anon": False} in opened
    assert delivery  # the module under test is the one the runner delivered through
