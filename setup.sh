#!/bin/bash

# Setting up the project
sudo dnf install julia
git clone https://github.com/NFSturm/Symmachus.jl.git
cd Symmachus.jl
julia setup_script.jl # Instantiating packages

# Running the model
julia src/SymmachusModel.jl
