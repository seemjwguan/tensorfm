# Welcome!

This repository contains the implementations of tensor factorization machines (FMs) studied in our paper "*Tensor factorization machine and its lifted form*."

# How to use?

To reproduce our results:

1. Download the datasets used in our paper from [here](https://drive.google.com/file/d/1_oGvvwjebGbKSODzHqtw7jU0GAp_9J_K/view?usp=sharing). (Due to size limit of GitHub, we are unable to upload them to this repository; you may also use whatever dataset else you want.)
2. Run *tensorFM_handler_ori.m* for tensor FM whereas *tensorFM_handler_lifted.m* for lifted tensor FM to estimate parameters.
3. 

# Dependencies

The following packages are required for running the programs:

- MATLAB Tensor Toolbox (downloadable from [here](https://www.tensortoolbox.org/)).
- MATLAB Parallel Computing Toolbox (not necessary, but if you want to use parallel computations for acceleration).
