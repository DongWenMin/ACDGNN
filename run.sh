#!/bin/bash
module load cuda/10.0
source activate dwmpy37
# python --version
python DSRADDI.py --epoch 30 --batch_size 2048 --ad 0.3 --fd 0.3 -s 'inductive1/data_0'\
 -aw 0.003 -lr 0.0007 -dsf true -e_dim 32 -r_dim 32 -hs 2 -cdt true