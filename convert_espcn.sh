#!/usr/bin/env bash

# run_y_only.sh
bash ./convert.sh espcn 1
bash ./convert.sh espcn_nc 1

# run_bmodel.sh
bash ./convert.sh espcn_ex
bash ./convert.sh espcn_cp
