#!/usr/bin/env bash
set -euo pipefail

echo "Descargando Titanic dataset..."
mkdir -p titanic

curl -sL "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic/train.csv" -o titanic/train.csv
curl -sL "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic/test.csv" -o titanic/test.csv
curl -sL "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic/gender_submission.csv" -o titanic/gender_submission.csv

echo "Dataset descargado en titanic/"
