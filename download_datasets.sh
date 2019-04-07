#!/bin/bash

wget https://zenodo.org/record/1161203/files/data.tar.gz
tar -zxvf data.tar.gz
rm -rf data/mnist/
rm -rf data/cifar10/
rm data.tar.gz
mkdir checkpoint
