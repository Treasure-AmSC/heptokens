"""Tests that each CLI entry point exits cleanly with --help and -h."""

import subprocess
import sys

import pytest

COMMANDS = [
    "heptokens-train",
    "heptokens-export",
    "heptokens-fit-preprocessors",
]

EXPECTED_FRAGMENTS = {
    "heptokens-train": ["heptokens-train", "datamodule=<NAME>", "Hydra options"],
    "heptokens-export": ["heptokens-export", "ckpt_path=<PATH>", "Hydra options"],
    "heptokens-fit-preprocessors": [
        "heptokens-fit-preprocessors",
        "datamodule=<NAME>",
        "Hydra options",
    ],
}


@pytest.mark.parametrize("cmd", COMMANDS)
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_exits_zero(cmd, flag):
    result = subprocess.run(
        [sys.executable, "-m", f"heptokens.{_cmd_to_module(cmd)}", flag],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`{cmd} {flag}` exited {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize("cmd", COMMANDS)
def test_help_content(cmd):
    result = subprocess.run(
        [sys.executable, "-m", f"heptokens.{_cmd_to_module(cmd)}", "--help"],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    for fragment in EXPECTED_FRAGMENTS[cmd]:
        assert fragment in output, (
            f"`{cmd} --help` output missing expected text '{fragment}'.\nOutput:\n{output}"
        )


def _cmd_to_module(cmd: str) -> str:
    return {
        "heptokens-train": "train",
        "heptokens-export": "export_tokens",
        "heptokens-fit-preprocessors": "fit_preprocessors",
    }[cmd]
