#!bin/bash

set -e

GREEN="\033[1;32m"
CYAN="\033[1;36m"
RESET="\033[0m"

echo -e "${CYAN} 1/2: Preprocessing test...${RESET}"
pytest src/models/test_preprocess.py

echo -e "${CYAN} 2/2: Training test...${RESET}"
pytest src/models/test_train.py

echo -e "${GREEN} Tests are done.${RESET}"
