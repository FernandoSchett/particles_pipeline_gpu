#!/bin/bash

VENV_DIR="$HOME/gsp25_particle_sfc"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
