"""Opt-in developer check: can a researcher follow the getting-started guide?

It copies the repository files from disk (including uncommitted edits) into a
temporary folder, follows the getting-started install steps there in a new environment,
and runs ``demo_tracking.ipynb`` and ``demo_sweep.ipynb`` top to bottom. Nothing
under the real home directory is changed. Run it with:

    .venv/bin/python -m pytest projects/turboswan/tests -o addopts='' -m developer -q

Set TURBOSWAN_KEEP_DEV_ENV=1 to keep the temporary folder for inspection.
"""

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.developer

REPO_ROOT = Path(__file__).parents[3]
NOTEBOOKS = ["demo_tracking.ipynb", "demo_sweep.ipynb"]


def run(command, *, cwd, env, timeout):
    """Run a command in its own process group so a timeout also stops its children."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"timed out after {timeout}s: {' '.join(command)}\n{output[-3000:]}")
    finally:
        # Kernel and Ray processes started by the command must not outlive it.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.returncode != 0:
        pytest.fail(f"failed ({process.returncode}): {' '.join(command)}\n{output[-3000:]}")
    return output


@pytest.fixture
def fresh_workspace(e2e_data_dir):
    """Yield (root, repo, env, uv): a temporary copy of the repo with an isolated home.

    The temporary folder is removed afterwards unless TURBOSWAN_KEEP_DEV_ENV is set.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required")
    real_env = os.environ
    # Short path: Ray's sockets fail on long temporary directories.
    root = Path(tempfile.mkdtemp(prefix="tsw-"))
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=REPO_ROOT, capture_output=True, check=True,
        ).stdout.decode().split("\0")
        repo = root / "repo"
        for name in filter(None, listed):
            source = REPO_ROOT / name
            if source.is_file():
                (repo / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, repo / name)

        home = root / "home"
        data_link = home / "turboswan" / "data" / "demo"
        data_link.parent.mkdir(parents=True)
        data_link.symlink_to(e2e_data_dir, target_is_directory=True)

        def query(*args):
            return subprocess.run(
                [uv, *args], capture_output=True, text=True, check=True
            ).stdout.strip()

        # A minimal environment: only what a researcher's shell would provide.
        env = {
            "HOME": str(home),
            "PATH": f"{Path(uv).parent}:/usr/local/bin:/usr/bin:/bin",
            "UV_CACHE_DIR": real_env.get("UV_CACHE_DIR") or query("cache", "dir"),
            "UV_PYTHON_INSTALL_DIR": query("python", "dir"),
            "RAY_TMPDIR": str(root / "ray"),
        }
        yield root, repo, env, uv
    finally:
        if os.environ.get("TURBOSWAN_KEEP_DEV_ENV"):
            print(f"\nkept developer-check workspace: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def test_readme_install_runs_demo_notebooks(fresh_workspace):
    """Following the getting-started guide in a new environment lets both demo notebooks run to the end.

    The notebooks pick their kernel from their own metadata (`turboswan`), so a wrong
    kernel name or an unregistered kernel fails here. The runner is a separate
    environment so that nothing outside the project's own `.venv` can satisfy imports.
    """
    root, repo, env, uv = fresh_workspace

    # The getting-started install steps.
    run([uv, "venv", "--python", "3.12"], cwd=repo, env=env, timeout=600)
    run([uv, "sync", "--package", "turboswan"], cwd=repo, env=env, timeout=1800)
    run(
        [
            ".venv/bin/python", "-m", "ipykernel", "install", "--user",
            "--name", "turboswan", "--display-name", "Python (turboswan)",
        ],
        cwd=repo, env=env, timeout=300,
    )

    # Notebook runner kept outside the project environment under test.
    runner = root / "runner"
    run([uv, "venv", "--python", "3.12", str(runner)], cwd=root, env=env, timeout=600)
    run(
        [uv, "pip", "install", "--python", str(runner / "bin" / "python"), "nbconvert", "ipykernel"],
        cwd=root, env=env, timeout=900,
    )

    notebooks = repo / "projects" / "turboswan" / "notebooks"
    for name in NOTEBOOKS:
        run(
            [
                str(runner / "bin" / "python"), "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--output-dir", str(root / "executed"),
                "--ExecutePreprocessor.timeout=1500", str(notebooks / name),
            ],
            cwd=notebooks, env=env, timeout=1800,
        )
        assert (root / "executed" / name).is_file()
