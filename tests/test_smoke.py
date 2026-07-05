import os
import subprocess


def test_readme_exists():
    assert os.path.isfile("README.md")


def test_gitignore_exists():
    assert os.path.isfile(".gitignore")


def test_csv_files_not_tracked():
    result = subprocess.run(
        ["git", "ls-files", "titanic/train.csv", "titanic/test.csv"],
        capture_output=True, text=True
    )
    assert result.stdout.strip() == "", f"CSV files still tracked: {result.stdout.strip()}"
