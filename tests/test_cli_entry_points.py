"""
Test that all CLI entry points can be imported and invoked with --help.

This validates the full import chain for each script (no DGL remnants,
no broken imports) without requiring datasets or GPU.
"""

import subprocess
import sys
import pytest


CLI_COMMANDS = [
    "qtaim-embed-train-graph",
    "qtaim-embed-train-node",
    "qtaim-embed-train-link",
    "qtaim-embed-train-graph-classifier",
    "qtaim-embed-bayes-opt-graph",
    "qtaim-embed-bayes-opt-node",
    "qtaim-embed-bayes-opt-graph-classifier",
    "qtaim-embed-mol2lmdb",
    "qtaim-embed-mol2lmdb-node",
    "qtaim-embed-data-summary",
]


@pytest.mark.parametrize("cmd", CLI_COMMANDS)
def test_cli_help(cmd):
    """Each CLI entry point should respond to --help without error."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--co", "-q"],  # dummy, replaced below
        capture_output=True,
        text=True,
    )
    # Actually run the CLI command with --help
    result = subprocess.run(
        [cmd, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{cmd} --help failed with code {result.returncode}.\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "usage:" in result.stdout.lower() or "optional arguments" in result.stdout.lower(), (
        f"{cmd} --help did not produce expected help output.\n"
        f"stdout: {result.stdout[:500]}"
    )


SCRIPT_MODULES = [
    "qtaim_embed.scripts.train.train_qtaim_graph",
    "qtaim_embed.scripts.train.train_qtaim_node",
    "qtaim_embed.scripts.train.train_qtaim_link",
    "qtaim_embed.scripts.train.train_qtaim_graph_classifier",
    "qtaim_embed.scripts.train.bayes_opt_graph",
    "qtaim_embed.scripts.train.bayes_opt_node",
    "qtaim_embed.scripts.train.bayes_opt_graph_classifier",
    "qtaim_embed.scripts.helpers.mol2lmdb",
    "qtaim_embed.scripts.helpers.mol2lmdb_node",
    # data_summary calls main() at module level (no __name__ guard),
    # so it cannot be import-tested. Covered by test_cli_help instead.
]


@pytest.mark.parametrize("module", SCRIPT_MODULES)
def test_script_importable(module):
    """Each script module should be importable and have a main() function."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}; assert hasattr({module}, 'main')"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Failed to import {module} or missing main().\n"
        f"stderr: {result.stderr[:500]}"
    )
