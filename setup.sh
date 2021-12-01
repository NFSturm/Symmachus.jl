#!/bin/bash

# Setting up the project
sudo dnf install julia git neovim unzip zip

git clone https://github.com/NFSturm/Symmachus.jl.git
cd Symmachus.jl
julia setup_script.jl # Instantiating packages

mkdir -p data/labels cache/final_model search_cache
