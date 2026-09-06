#!/bin/bash

# clone satlaspretrain
mkdir -p ../../external/satlaspretrain
git clone https://github.com/allenai/satlaspretrain_models.git ../../external/satlaspretrain

cd ../../external/satlaspretrain
# checkout a specific commit
git checkout 7b5cd45adc3cad70b3834d65956974af6f6bffd0
cd -

# satlaspretrain_models' own code does absolute self-imports (e.g. `from
# satlaspretrain_models.utils import ...`), so external/satlaspretrain needs to be on
# PYTHONPATH directly (not just importable via epimask.external.satlaspretrain). Its
# setup.py requires Python >=3.9, which is newer than this repo's environment.yml
# (Python 3.8), so `pip install -e` is not an option here -- scripts/epimask/*.sh add
# external/satlaspretrain to PYTHONPATH for this reason.