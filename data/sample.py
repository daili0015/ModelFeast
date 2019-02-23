#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-18 19:49:08
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-18 20:15:44

import pandas as pd 

sam = pd.read_csv('/home/DL/ModelFeast/data/submit_example.csv')
print(sam.dtypes)
print(sam.head())
