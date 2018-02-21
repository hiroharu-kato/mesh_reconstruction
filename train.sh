#!/usr/bin/env bash

# multi class
python mesh_reconstruction/train.py -eid multiclass -li 100000 -ni 1000000

# single class with smoothness loss
python mesh_reconstruction/train.py -eid singleclass_02691156 -cls 02691156 -ls 0.001 -li 1000 -ni 200000
python mesh_reconstruction/train.py -eid singleclass_02828884 -cls 02828884 -ls 0.001 -li 1000 -ni 100000
python mesh_reconstruction/train.py -eid singleclass_02933112 -cls 02933112 -ls 0.001 -li 1000 -ni 40000
python mesh_reconstruction/train.py -eid singleclass_02958343 -cls 02958343 -ls 0.001 -li 1000 -ni 50000
python mesh_reconstruction/train.py -eid singleclass_03001627 -cls 03001627 -ls 0.001 -li 1000 -ni 250000
python mesh_reconstruction/train.py -eid singleclass_03211117 -cls 03211117 -ls 0.001 -li 1000 -ni 20000
python mesh_reconstruction/train.py -eid singleclass_03636649 -cls 03636649 -ls 0.001 -li 1000 -ni 20000
python mesh_reconstruction/train.py -eid singleclass_03691459 -cls 03691459 -ls 0.001 -li 1000 -ni 40000
python mesh_reconstruction/train.py -eid singleclass_04090263 -cls 04090263 -ls 0.001 -li 1000 -ni 130000
python mesh_reconstruction/train.py -eid singleclass_04256520 -cls 04256520 -ls 0.001 -li 1000 -ni 40000
python mesh_reconstruction/train.py -eid singleclass_04379243 -cls 04379243 -ls 0.001 -li 1000 -ni 90000
python mesh_reconstruction/train.py -eid singleclass_04401088 -cls 04401088 -ls 0.001 -li 1000 -ni 30000
python mesh_reconstruction/train.py -eid singleclass_04530566 -cls 04530566 -ls 0.001 -li 1000 -ni 90000
