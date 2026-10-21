"""Console deployment files: the sidecar image, compose services and frontend CI (womblex.ui)."""
from __future__ import annotations

import argparse
import json
import re

import pytest
import yaml

pytest.importorskip("fastapi")

from tests._ui_helpers import REPO_ROOT
from womblex.cli import ALL_COMMANDS


def _compose() -> dict:
    return yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())


def _compose_service(name: str) -> dict:
    return _compose()["services"][name]

class TestSidecarImage:
    """The console container is only correct if it agrees with the CLI and the app.

    Nothing else in the suite reads the deployment files, so a renamed flag or
    a second run source would otherwise surface as a container that exits 1 in
    someone's compose stack.
    """

    def test_entrypoint_parses_against_the_real_cli(self) -> None:
        """`womblex ui`'s flags are the image's entrypoint — they must still exist."""
        match = re.search(
            r"^ENTRYPOINT (\[.*\])$", (REPO_ROOT / "Dockerfile.ui").read_text(), re.MULTILINE
        )
        assert match, "Dockerfile.ui has no JSON-form ENTRYPOINT"
        entrypoint = json.loads(match.group(1))
        assert entrypoint[0] == "womblex"

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        for cmd in ALL_COMMANDS:
            cmd.register(sub.add_parser(cmd.name))
        args = parser.parse_args(entrypoint[1:])  # SystemExit if a flag is gone
        assert args.command == "ui"
        # Loopback is the CLI default, and inside a container it would refuse
        # every request arriving via the published port.
        assert args.host == "0.0.0.0"

    def test_compose_ui_service_is_read_only_with_tmpfs(self) -> None:
        ui = _compose_service("ui")
        # The console image, named rather than built: the base file pins a
        # published image and the local override carries the build. Which
        # service runs which image is audited in docs/deployment-images.md.
        assert "womblex-console" in ui["image"]
        assert ui["read_only"] is True
        # A store-backed read stages the manifest through a temp dir.
        assert "/tmp" in ui["tmpfs"]

    def test_compose_ui_service_resolves_exactly_one_run_source(self) -> None:
        """Two sources make resolve_settings raise, so the container would exit 1."""
        env = _compose_service("ui")["environment"]
        assert "WOMBLEX_STORE_URI" in env
        assert "WOMBLEX_UI_OUTPUT_ROOT" not in env

    def test_compose_ui_service_waits_for_the_queue_and_store_it_advertises(self) -> None:
        """The ui service advertises WOMBLEX_DB_DSN and WOMBLEX_STORE_URI, so it must
        wait for postgres AND minio to be *healthy* — not just for the bucket to
        exist. Coming up before either is ready was the cause of the reported
        "queue unreachable" and the intermittent Composer 500s/failed-to-fetch:
        every screen's first reads raced the backends' startup.
        """
        deps = _compose_service("ui")["depends_on"]
        env = _compose_service("ui")["environment"]
        # It advertises both, so a consumer expects both to answer on boot.
        assert "WOMBLEX_DB_DSN" in env and "WOMBLEX_STORE_URI" in env
        assert deps["postgres"]["condition"] == "service_healthy"
        assert deps["minio"]["condition"] == "service_healthy"
        # The bucket must also exist before the store reads land.
        assert deps["createbuckets"]["condition"] == "service_completed_successfully"

    def test_bundled_backends_sit_behind_the_local_profile(self) -> None:
        """The self-contained stack (bundled Postgres/MinIO/bucket) is opt-in via the
        `local` profile, so a cloud deployment against external Postgres + S3 does
        not start them: `docker compose up worker` brings up only the profile-less
        services. Without this, cloud mode would spin up a Postgres/MinIO nobody
        uses (and, worse, `worker`'s deps would wait on them).
        """
        svcs = _compose()["services"]
        for name in ["postgres", "minio", "createbuckets"]:
            assert svcs[name]["profiles"] == ["local"], name
        # The consumers carry no profile — they are the default `up` target.
        for name in ["init", "worker", "ui"]:
            assert "profiles" not in svcs[name], name

    def test_cloud_services_do_not_hard_depend_on_the_bundled_backends(self) -> None:
        """Every dependency a cloud service declares on a `local`-profile backend is
        `required: false`, so an absent bundled Postgres/MinIO/bucket (the cloud
        case, external services instead) is a warning, not a missing-dependency
        error. This is the one thing that lets `docker compose up worker` run at
        all when the bundled backends were never started.
        """
        svcs = _compose()["services"]
        local_backends = {"postgres", "minio", "createbuckets", "init"}
        for name in ["init", "worker", "womblex", "seed-demo", "ui"]:
            for dep, spec in svcs[name].get("depends_on", {}).items():
                if dep in local_backends:
                    assert spec["required"] is False, f"{name} hard-depends on {dep}"

    def test_ui_health_gate_survives_into_cloud_mode(self) -> None:
        """The startup-race fix (ui waits for postgres/minio *healthy*) must still
        hold as `required: false` — the condition is unchanged, so a local stack
        still gates on health, and a cloud stack simply has no bundled backend to
        wait on (it reads the external DSN/S3 the same env names).
        """
        deps = _compose_service("ui")["depends_on"]
        assert deps["postgres"]["condition"] == "service_healthy"
        assert deps["postgres"]["required"] is False
        assert deps["minio"]["condition"] == "service_healthy"
        assert deps["minio"]["required"] is False

    def test_connection_env_is_overridable_with_bundled_local_defaults(self) -> None:
        """The whole local/cloud switch: each connection var reads from the
        environment with the bundled local stack as the `${VAR:-default}`
        fallback. Unset env == today's local stack (the defaults point at the
        bundled services); set env == external Postgres + S3. A hard-coded value
        here would silently ignore an operator's external endpoint.

        Store credentials live on the store-specific `WOMBLEX_S3_ACCESS_KEY_ID` /
        `WOMBLEX_S3_SECRET_ACCESS_KEY` (defaulting to MinIO's key locally), while
        the ambient `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` default to
        **empty** on purpose: setting them is process-global and would clobber
        the EC2 instance role the isaacus-sagemaker SigV4 signer resolves in a
        cloud deployment (the credential-conflict bug).
        """
        raw = (REPO_ROOT / "docker-compose.yml").read_text()
        # The connection vars + store creds are all `${VAR:-<bundled default>}`.
        for var, default in [
            # The bundled compose default, already baselined in docker-compose.yml.
            ("WOMBLEX_DB_DSN", "postgresql://womblex:womblex@postgres:5432/womblex"),  # pragma: allowlist secret
            ("WOMBLEX_STORE_URI", "s3://womblex"),
            ("WOMBLEX_S3_ENDPOINT", "http://minio:9000"),
            ("WOMBLEX_S3_ACCESS_KEY_ID", "minioadmin"),
            ("WOMBLEX_S3_SECRET_ACCESS_KEY", "minioadmin"),
            ("AWS_REGION", "us-east-1"),
        ]:
            assert f"${{{var}:-{default}}}" in raw, var
        # The AWS credential vars pass through only when the operator sets them,
        # defaulting to empty so the SageMaker signer can reach the instance role.
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            assert f"${{{var}:-}}" in raw, var
        # The anchor the services inherit carries the same overridable form —
        # `yaml.safe_load` does not run compose's substitution, so the literal
        # `${VAR:-default}` string is what a service's environment shows.
        env = _compose_service("worker")["environment"]
        assert env["WOMBLEX_DB_DSN"] == (
            "${WOMBLEX_DB_DSN:-postgresql://womblex:womblex@postgres:5432/womblex}"  # pragma: allowlist secret
        )
        assert env["WOMBLEX_STORE_URI"] == "${WOMBLEX_STORE_URI:-s3://womblex}"
        assert env["WOMBLEX_S3_ENDPOINT"] == "${WOMBLEX_S3_ENDPOINT:-http://minio:9000}"
        assert env["WOMBLEX_S3_ACCESS_KEY_ID"] == "${WOMBLEX_S3_ACCESS_KEY_ID:-minioadmin}"
        assert env["AWS_ACCESS_KEY_ID"] == "${AWS_ACCESS_KEY_ID:-}"

    def test_frontend_builder_stage_runs_a_real_package_script(self) -> None:
        """The builder stage's `npm run build` must name a script `ui/package.json` declares."""
        dockerfile = (REPO_ROOT / "Dockerfile.ui").read_text()
        match = re.search(r"^RUN npm run (\S+)$", dockerfile, re.MULTILINE)
        assert match, "Dockerfile.ui has no `npm run <script>` build step"
        scripts = json.loads((REPO_ROOT / "ui" / "package.json").read_text())["scripts"]
        assert match.group(1) in scripts

    def test_final_stage_copies_the_builder_stages_declared_output(self) -> None:
        """The adapter output dir in `vite.config.ts` must match what the Dockerfile COPYs out."""
        vite_config = (REPO_ROOT / "ui" / "vite.config.ts").read_text()
        assert "pages: 'build'" in vite_config
        dockerfile = (REPO_ROOT / "Dockerfile.ui").read_text()
        assert "COPY --from=frontend-builder /app/ui/build /app/ui/build" in dockerfile


class TestFrontendCi:
    """`ui/`'s CI job must call scripts that still exist — otherwise a rename
    here silently stops linting or building the SPA.
    """

    def test_frontend_job_runs_declared_package_scripts(self) -> None:
        workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
        frontend = workflow["jobs"]["frontend"]
        scripts = json.loads((REPO_ROOT / "ui" / "package.json").read_text())["scripts"]
        run_steps = [
            step["run"] for step in frontend["steps"]
            if step.get("working-directory") == "ui" and "run" in step
        ]
        assert run_steps, "no step in the frontend job runs anything in ui/"
        for run in run_steps:
            match = re.match(r"npm (run )?(\S+)$", run.strip())
            assert match, f"unrecognised frontend CI step: {run!r}"
            script = match.group(2)
            if script != "ci":  # `npm ci` installs; it isn't a package.json script
                assert script in scripts, f"ui/package.json has no {script!r} script"
