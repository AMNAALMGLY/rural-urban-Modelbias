#!/bin/bash

fold=('A' 'B' 'C' 'D' 'E')
#bands=('ms' 'merge' )
#ckpts=('' '' '' '' '')

for f in "${fold[@]}"; do
    #for f in "${fold[@]}"; do
    sbatch ./train_.sh --fold=$f
    #done
done

