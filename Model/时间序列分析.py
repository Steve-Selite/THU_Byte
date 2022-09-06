# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:15:19 2022

@author: lenovo
"""

import pandas as pd
from arch import arch_model, univariate
import datetime as dt
import numpy as np
import math
import time
from arch.unitroot import ADF
from statsmodels.tsa import stattools  # 白噪声检验:Ljung-Box检验
from statsmodels.tsa.arima.model import ARIMA #导入ARIMA模型
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from collections import defaultdict
pandas2ri.activate()
from rpy2.robjects import globalenv
importr("rugarch")

# 数据预处理
def data_process(n, fund, index, rel):
    pos = list(fund.iloc[1, :].values).index(n)
    fund_value = fund.iloc[3:245, pos].to_frame()
    index_name = rel[rel['证券名称'] == n]['主要跟踪标的代码\n第1名'].values
    pos1 = list(index.iloc[0, :].values).index(index_name)
    index_value = index.iloc[3:304, pos1].to_frame()
    fund_value.columns = ['基金收盘价']
    index_value.columns = ['基准收盘价']
    data = fund_value.join(index_value, how='inner')
    data['基金收盘价的log'] = data['基金收盘价'].astype('float').apply(np.log)
    data['基准收盘价的log'] = data['基准收盘价'].astype('float').apply(np.log)
    data['基金收益率'] = data['基金收盘价的log'].diff() * 100
    data['基准收益率'] = data['基准收盘价的log'].diff() * 100
    return data.iloc[1:,-2:]

# 数据时间序列检验
def check(data):
    adf_fund = ADF(data['基金收益率'])
    adf_index = ADF(data['基准收益率'])
    if adf_fund.pvalue < 0.1 and adf_index.pvalue < 0.1:
        LjungBox_fund = stattools.q_stat(stattools.acf(data['基金收益率']), len(data['基金收益率']))[1][-1]  # 显示第一个到第11个白噪声检验的p值
        LjungBox_index = stattools.q_stat(stattools.acf(data['基准收益率']), len(data['基准收益率']))[1][-1]  # 显示第一个到第11个白噪声检验的p值
        if LjungBox_fund < 0.1 and LjungBox_index < 0.1:
            model = arch_model(y=data['基金收益率'], x=data['基准收益率'], mean='LS')  # 白噪声检验通过，直接确定模型
            result = model.fit()
            resid1 = result.resid  # 提取残差
            LjungBox1 = stattools.q_stat(stattools.acf(resid1 ** 2), len(resid1))[1][-1]  # 残差平方序列的白噪声检验
            if LjungBox1 < 0.1:# 拒绝原假设，则残差序列具有ARCH效应
                return 1,1
            else:
                return 1,0
    return 0,0

if __name__ == "__main__":
    name = pd.read_excel('../Data/fund_name.xlsx')
    fund_name = name['基金名称'].values
    fund = pd.read_excel('../Data/基金净值.xlsx')
    index = pd.read_excel('../Data/指数净值.xlsx')
    rel = pd.read_excel('../Data/基金-指数对应关系.xlsx')
    date = ['2022-08-09'] #'2021-12-31','2021-09-30','2022-03-31','2022-06-30',
    results_piaoyi = defaultdict(list)
    for n in fund_name:
        data = data_process(n, fund, index, rel)
        start = 0
        for d in date:
            datetime = list(data.index)
            time_stamp = np.datetime64(d)
            pos = datetime.index(time_stamp)
            new_data = data.iloc[start:pos+1,]
            start = pos+1
            ADF_flag, ARCH_flag = check(new_data)
            if ADF_flag == 1:
                if ARCH_flag == 0:
                    arx = univariate.ARX(y=new_data['基金收益率'], x=new_data['基准收益率'], lags=[1])
                    res = arx.fit()
                    if res._params[1] < 0:
                        results_piaoyi[d].append([n, res._params[1], np.nan, '是'])
                    else:
                        results_piaoyi[d].append([n, res._params[1], np.nan, '否'])
                else:
                    am1 = arch_model(y=new_data['基准收益率'])
                    res1 = am1.fit(update_freq=5)
                    v = res1.conditional_volatility
                    data1 = pd.concat([new_data, v], axis=1)
                    data1.columns = ['fund', 'index', 'var']
                    data1.index = list(data1.index)
                    y = pandas2ri.py2rpy(data1)
                    globalenv['y'] = y
                    rscript = """
                            myspec<-ugarchspec(variance.model = list(model="eGARCH",garchOrder=c(1,1),submodel=NULL,
                                                                    external.regressors=matrix(y$var),variance.targeting=FALSE),
                                              mean.model =list(armaOrder = c(1,0), include.mean=TRUE,archm=TRUE,archpow=1,
                                                               arfima=FALSE,external.regressors=matrix(y$index),archex=FALSE))
                            myfit<-ugarchfit(myspec,data=y$fund)
                            xishu<-coef(myfit)
                            """#
                    results = r(rscript)
                    if results[3] > 0 and results[-1] > 0:
                        results_piaoyi[d].append([n, results[3], results[-1], '否'])
                    else:
                        results_piaoyi[d].append([n, results[3], results[-1], '是'])
    writer = pd.ExcelWriter('../result/timeseries.xlsx')
    for d in date:
        result = pd.DataFrame(results_piaoyi[d], columns=['基金名称', '收益率维度', '风险维度', '是否漂移'])
        result.to_excel(excel_writer=writer, sheet_name=d)
    writer.save()
    writer.close()