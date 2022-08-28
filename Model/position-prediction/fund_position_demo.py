# -*- coding: utf-8 -*-
#作为函数调用接口API

from fund_position_func import Algo_fund_position

Algo_fund_position(result_dir='./results/ridge').main(model_type = 'ridge',path = './data/name.xlsx', each_len = 24, PCA_n = 9, lamda = 0.001)
