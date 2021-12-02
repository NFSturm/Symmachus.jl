#!/bin/bash

# Setting up the project
sudo dnf install julia neovim unzip zip

cd Symmachus.jl
julia setup_script.jl # Instantiating packages

mkdir -p data/labels cache/final_model search_cache
