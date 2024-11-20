#!bin/bash

set -e

GREEN="\033[1;32m"
CYAN="\033[1;36m"
RESET="\033[0m"

echo -e "${CYAN} 1/2: Preprocessing data...${RESET}"
python3 src/models/preprocess_data.py

echo -e "${CYAN} 2/2: Training data...${RESET}"
python3 src/models/train_data.py

echo -e "${GREEN} Pipeline executed ${RESET}"
