#!/usr/bin/env bash

if [ -e "./data/mesh_reconstruction_models.zip" ]; then
    unzip ./data/mesh_reconstruction_models.zip -d ./data/
else
    echo "Please download dataset from https://drive.google.com/open?id=1tRHQoc0VWpj61PM1tVozIFwrFsDpKbTQ, put it in ./data/mesh_reconstruction_models.zip, and run this script."
fi
