"""Utility functions for handling conda virtual environments"""

from os.path import isdir, join, pardir
from subprocess import check_call
from sys import exit as sys_exit


VENV_DIR = ".venv"
ENVIRONMENT_FILE_NAME = "environment.yml"


def ensure_venv_installed():
    """Ensure that the conda environment is installed"""
    if not isdir(join(pardir, VENV_DIR)):
        raise RuntimeError("Virtual environment is not installed!")


def ensure_venv_not_installed():
    """Ensure that the conda environment is not installed"""
    if isdir(join(pardir, VENV_DIR)):
        print("Virtual environment is already set up!")
        sys_exit()


def setup_venv():
    """Set up the conda environment"""
    check_call(
        ["conda", "env", "create", "-f", ENVIRONMENT_FILE_NAME, "--prefix", join(pardir, ".venv")]
    )


def update_venv():
    """Update the conda environment"""
    check_call(
        [
            "conda",
            "env",
            "update",
            "--prefix",
            join(pardir, VENV_DIR),
            "--file",
            ENVIRONMENT_FILE_NAME,
            "--prune",
        ]
    )
