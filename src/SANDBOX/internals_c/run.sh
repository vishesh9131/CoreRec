#!/bin/bash

# Navigate to the directory containing the files
cd "$(dirname "$0")"

# Compile gat.c to gat.so
gcc -shared -o gat.so -fPIC gat.c
if [ $? -ne 0 ]; then
    echo "Failed to compile gat.c"
    exit 1
fi

# Compile gtransformer.c to graph_transformer.so
gcc -shared -o graph_transformer.so -fPIC gtransformer.c
if [ $? -ne 0 ]; then
    echo "Failed to compile gtransformer.c"
    exit 1
fi

# Run the Python script
python interface.py
if [ $? -ne 0 ]; then
    echo "Failed to run interface.py"
    exit 1
fi

echo "All tasks completed successfully."