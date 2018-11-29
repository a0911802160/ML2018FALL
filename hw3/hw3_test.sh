#!/bin/bash
wget -O best_model.h5 'https://www.dropbox.com/s/oqbpqmreeyt3i2p/best_model.h5?dl=0'

python3 hw3_test.py $1 $2