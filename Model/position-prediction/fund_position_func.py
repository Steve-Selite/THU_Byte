# -*- coding: utf-8 -*-

from dataProcess_Func import DataProcess
from Output_Func import Output
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from scipy.optimize import minimize
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from collections import defaultdict

class Algo_fund_position():
    '''
    预测基金仓位
    '''

    def __init__(self, result_dir):
        '''
        :param result_dir string: 存放结果文件的文件夹
        '''

        self.result_dir = result_dir

    def linear_model_fund(self, industry_profit, resudial, upper_bound, lamda, lamda_flag):
        '''
        线性回归最优解

        :param A1 numpy数组：线性回归的矩阵
        :param b1 numpy数组：线性回归的残差
        :param ub float：自变量上限
        :param lamda float：lasso回归的lamda值
        :param lamda_flag bool：是否使用lasso回归
        :return result numpy数组：目标函数值
        :return result1 float：约束值
        :return res1.x numpy数组：线性回归的解
        '''

        num_data = np.shape(industry_profit)[1]

        def my_func(data):
            ls = (resudial - np.dot(industry_profit, data)) ** 2
            result = np.sum(ls)
            if lamda_flag:
                result += lamda * np.sum(data ** 2)
            return result

        def constrain_u1(data):
            result1 = upper_bound - np.sum(data)
            return result1

        cons = ({'type': 'eq', 'fun': constrain_u1})
        data1 = [1 / num_data] * num_data
        bnds = []
        for i in range(num_data):
            bnds.append((0, 0.95))
        res1 = minimize(my_func, data1, bounds=bnds, constraints=cons)
        return res1.x

    def PCA_(self, funds, n):
        '''
        PCA降维

        :param funds numpy数组：回归自变量
        :param n float：PCA保留主成分的百分比
        :return: fund1 numpy数组：PCA降维后的自变量
        :return: pca.components_ numpy数组：PCA用于降维的数组
        '''

        pca = PCA(n_components=n)
        fund = pca.fit(funds)
        fund1 = pca.transform(funds)
        return fund1, pca.components_

    def cal_ave(self, result):
        '''
        计算平均值

        :param result dataframe：结果数据
        :return: result dataframe：加上行列总平均值的数据
        '''

        ave = []
        sum_all = 0
        for i in range(len(result[0]) - 1):
            sum1 = 0
            for j in range(len(result)):
                sum1 += result[j][i]
            sum_all += sum1
            ave.append(sum1 / len(result))
        ave.append(sum_all / (len(result) * (len(result[0]) - 1)))
        result.append(ave)
        return result

    def judge_type(self, res, index_name):
        maxres = max(res)
        if maxres < 0.1:
            return '行业混合型'
        else:
            res_pos = np.argsort(-res)
            return index_name[res_pos[0]]+','+index_name[res_pos[1]]+','+index_name[res_pos[2]]

    def linear_model(self, datelist, fundlist, fund_name, index_name, date_find, fund, index, position, each_len, output, lamda, lamda_flag, PCA_n, PCA_flag):
        '''
        采用线性模型（即OLS/OLS+PCA/lasso回归进行仓位预测）

        :param datelist list：日期列表
        :param fundlist list：所预测基金的名字列表
        :param fund_name list：所有基金的名字列表
        :param index_name list：行业的名字列表
        :param date_find list：每季度最后一个交易日在datelist中对应的位置
        :param fund dataframe：index为时间，columns为基金code的基金收益率数据
        :param index list：index为时间，columns为基金code的行业收益率数据
        :param position dict：key为日期，value为二维list，对应每个基金在该日的实际仓位
        :param each_len int：滚动长度（一月）
        :param output class：输出类
        :param lamda float：lasso回归的lamda
        :param lamda_flag bool：是否使用lasso回归
        :param PCA_n float：PCA的系数
        :param PCA_flag bool：是否使用PCA降维
        :return: res_final dict：key为日期，value为二维list，对应每个基金在该日的预测仓位
        :return: MSE1 list：预测的MSE
        :return: R1 list：以季度为节点回归的R平方
        :return: R2 list：滚动区间预测的R平方
        :return: adjustR1 list：以季度为节点回归的调整R平方
        :return: adjustR2 list：以季度为节点回归的调整R平方
        '''
        R1 = []
        adjustR1 = []
        R2 = []
        adjustR2 = []
        MSE1 = []
        fund_type = []
        res_final = defaultdict(pd.DataFrame)
        for date in datelist:
            p1 = datelist.index(date)
            res1 = []
            MSE3=[]
            R3=[]
            adjustR3=[]
            for name in fundlist:
                print(date+'-'+name)
                pos = fund_name.index(name)
                fund_pchg = fund.iloc[date_find[p1]:date_find[p1 + 1], pos]
                X_pchg = index.iloc[date_find[p1]:date_find[p1 + 1], :]
                N = len(fund_pchg.index)
                K = len(X_pchg.columns)
                #print(fund_pchg)
                #print(X_pchg)

                if PCA_flag:
                    X_pchg, pca_feature = self.PCA_(funds=X_pchg.to_numpy(), n=PCA_n)

                #initial = position[datelist[p1 - 1]][fundlist.index(name)]
                ub = 1
                position_predict = self.linear_model_fund(industry_profit=X_pchg, resudial=fund_pchg.to_numpy(), upper_bound=ub, lamda=lamda, lamda_flag=lamda_flag)
                fund_pchg_pre = np.dot(X_pchg, position_predict)
                if PCA_flag:
                    position_predict = np.dot(position_predict, pca_feature)
                    position_predict[position_predict<0]=0
                result = position_predict / sum(position_predict)
                fund_type.append([date, name, self.judge_type(result, index_name)])
                output.plot('predict & real fund position', date + "-" + name + ".jpg", result,
                            position[date][fundlist.index(name)], index_name)

                ave_pchg = abs(fund_pchg - np.mean(fund_pchg))
                diff = abs(result - position[date][fundlist.index(name)])
                diff_pchg = abs(fund_pchg - fund_pchg_pre)
                MSE = np.sum(diff ** 2) / len(diff)  # MSE规格，一个基金预测的MSE与行业平均MSE，R平方的标规
                R = 1 - np.sum(diff_pchg ** 2) / np.sum(ave_pchg ** 2)
                adjust_R = 1 - (1 - R) * (N - 1) / (N - 1 - K)
                MSE3.append(MSE)
                R3.append(R)
                adjustR3.append(adjust_R)
                res1.append(result)
            MSE3.append(sum(MSE3) / len(MSE3))
            R3.append(sum(R3) / len(R3))
            adjustR3.append(sum(adjustR3) / len(adjustR3))
            MSE1.append(MSE3)
            R1.append(R3)
            adjustR1.append(adjustR3)
            res_final[date] = res1

        '''datelist_all = list(fund.index)
        small_bound = 0
        big_bound = each_len
        while big_bound <= len(datelist_all):
            R4=[]
            adjustR4=[]
            for name in fundlist:
                pos = fund_name.index(name)
                fund_pchg = fund.iloc[small_bound:big_bound, pos]
                X_pchg = index.iloc[small_bound:big_bound, :]
                N = len(X_pchg.index)
                K = len(X_pchg.columns)
                #print(fund_pchg)
                #print(X_pchg)

                if PCA_flag:
                    X_pchg, pca_feature = self.PCA_(funds=X_pchg.to_numpy(), n=PCA_n)

                position_predict = self.linear_model_fund(industry_profit=X_pchg, resudial=fund_pchg.to_numpy(), upper_bound=ub, lamda=lamda, lamda_flag=lamda_flag)
                fund_pchg_pre = np.dot(X_pchg, position_predict)
                diff_pchg = abs(fund_pchg - fund_pchg_pre)
                ave_pchg = abs(fund_pchg - np.mean(fund_pchg))
                R = 1 - np.sum(diff_pchg ** 2) / np.sum(ave_pchg ** 2)
                adjust_R = 1 - (1 - R) * (N - 1) / (N - 1 - K)
                R4.append(R)
                adjustR4.append(adjust_R)
            R4.append(sum(R4) / len(R4))
            adjustR4.append(sum(adjustR4) / len(adjustR4))
            small_bound += 1
            big_bound += 1
            R2.append(R4)
            adjustR2.append(adjustR4)'''

        MSE1 = self.cal_ave(MSE1)
        R1 = self.cal_ave(R1)
        adjustR1 = self.cal_ave(adjustR1)
        #R2 = self.cal_ave(R2)
        #adjustR2 = self.cal_ave(adjustR2) R2,adjustR2,

        return res_final, MSE1, R1, adjustR1, fund_type

    def karma(self, datelist, fundlist, fund_name, index_name, date_find, fund, index, position, output, industry_flag):
        '''
        采用卡尔曼滤波模型进行仓位预测

        :param datelist list：日期列表
        :param fundlist list：所预测基金的名字列表
        :param fund_name list：所有基金的名字列表
        :param index_name list：行业的名字列表
        :param date_find list：每季度最后一个交易日在datelist中对应的位置
        :param fund dataframe：index为时间，columns为基金code的基金收益率数据
        :param index list：index为时间，columns为基金code的行业收益率数据
        :param position dict：key为日期，value为二维list，对应每个基金在该日的实际仓位
        :param output class：输出类
        :param industry_flag bool：是否使用行业指数
        :return: res_final dict：key为日期，value为二维list，对应每个基金在该日的预测仓位
        :return: MSE1 list：预测的MSE
        :return: R1 list：预测的R平方
        :return: adjustR1 list：预测的调整R平方
        '''

        MSE1 = []
        R1 = []
        adjustR1 = []
        fund_type = []
        res_final = defaultdict(list)
        for date in datelist[1:]:
            p1 = datelist.index(date)
            res1 = []
            MSE2 = []
            R2 = []
            adjustR2 = []
            for name in fundlist:
                print(date + '-' + name)
                pos = fund_name.index(name)
                fund_pchg = fund.iloc[date_find[p1 - 1]:date_find[p1], pos]
                if industry_flag:
                    X_pchg = index[name].iloc[date_find[p1 - 1]:date_find[p1], :]
                else:
                    X_pchg = index.iloc[date_find[p1 - 1]:date_find[p1], :]
                N = len(fund_pchg.index)
                K = len(X_pchg.columns)
                #print(fund_pchg)
                #print(X_pchg)
                measurements = []
                for i in fund_pchg.to_numpy():
                    measurements.append([i] * len(X_pchg.columns))
                measurements = np.array(measurements)
                T_matrix = [np.diag([1] * len(X_pchg.columns))] * len(X_pchg.index)
                O_matrix = []
                for i in X_pchg.to_numpy():
                    O_matrix.append(np.expand_dims(i, 0).repeat(i.shape[0], axis=0))
                initial = position[datelist[p1 - 1]][fundlist.index(name)]
                #print(position[datelist[p1 - 1]])
                #print(fundlist.index(name))
                #print(position[datelist[p1 - 1]][fundlist.index(name)])
                kf = KalmanFilter(transition_matrices=T_matrix, observation_matrices=np.array(O_matrix),
                                  initial_state_mean=initial)
                kf = kf.em(measurements, n_iter=5)
                (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
                state_predict = filtered_state_means[-1]
                state_predict[state_predict < 0] = 0
                result = state_predict / sum(state_predict)
                fund_type.append([date, name, self.judge_type(result, index_name)])
                fund_pchg_pre = np.dot(X_pchg, result)
                output.plot('predict & real fund position', date+"-"+name+".jpg", result, position[date][fundlist.index(name)], index_name)
                ave_pchg = abs(fund_pchg - np.mean(fund_pchg))
                diff = abs(result - position[date][fundlist.index(name)])
                diff_pchg = abs(fund_pchg - fund_pchg_pre)
                R = 1 - np.sum(diff_pchg ** 2) / np.sum(ave_pchg ** 2)
                adjust_R = 1 - (1 - R) * (N - 1) / (N - 1 - K)
                MSE = np.sum(diff ** 2) / len(diff)  # MSE规格，一个基金预测的MSE与行业平均MSE，R平方的标规
                MSE2.append(MSE)
                R2.append(R)
                adjustR2.append(adjust_R)
                res1.append(result)
                #Output_Plot
            #print(len(res1))
            MSE2.append(sum(MSE2)/len(MSE2))
            R2.append(sum(R2)/len(R2))
            adjustR2.append(sum(adjustR2)/len(adjustR2))
            MSE1.append(MSE2)
            R1.append(R2)
            adjustR1.append(adjustR2)
            res_final[date] = res1

        MSE1 = self.cal_ave(MSE1)
        R1 = self.cal_ave(R1)
        adjustR1 = self.cal_ave(adjustR1)

        return res_final, MSE1, R1, adjustR1, fund_type

    def main(self, model_type, path, each_len, PCA_n, lamda):
        '''
        主函数

        :param model_type str：选择的模型
        :param path str：保存输入的日期列表和基金名字
        :param each_len int：每段滚动的时间长度
        :param lamda float：lasso回归的lamda
        :param PCA_n float：PCA的系数
        '''
        datelist = pd.read_excel(path, sheet_name='日期列表')
        datelist = list(datelist['日期列表'].astype(str).values)
        fundlist = pd.read_excel(path, sheet_name='基金名字')
        fundlist = list(fundlist['基金列表'].astype(str).values)
        #print(datelist)
        #print(fundlist)
        fund, index, position, date_find, fund_name, index_name, index_position = DataProcess(data_dir='./data').main(
            data_fund=['/data-fund.xlsx', 'nav_adj公募基金复权单位净值', 'sector_index板块和申万一级行业'], col=[1, 6],
            data_industry='/行业.xlsx', datelist=datelist, fundlist=fundlist)
        output = Output(result_dir=self.result_dir)
        if model_type == 'karma':
            res_final, MSE, R1, adjustR1, fund_type = self.karma(datelist = datelist, fundlist = fundlist, fund_name = fund_name, index_name = index_name,
                                        date_find = date_find, fund = fund, index = index, position=position, output=output, industry_flag=False)
            output.csv_save(data = MSE, index1 = (datelist[1:]+['ave']), column = (fundlist+['ave']), path = 'MSE_result.xlsx')
            output.csv_save(data=R1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='R1_result.xlsx')
            output.csv_save(data=adjustR1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='adjustR1_result.xlsx')
            output.csv_sheet_save(data = res_final, path = 'predict_result.xlsx', index_name = index_name, fund_name = fundlist)
            fund_type_final = pd.DataFrame(fund_type, columns=['日期','基金','类型'])
            fund_type_final.to_excel(self.result_dir+'/基金类型预测结果.xlsx')
            #output_write
        elif model_type == 'karma_industry':
            res_final, MSE, R1, adjustR1, fund_type = self.karma(datelist=datelist, fundlist=fundlist, fund_name=fund_name,
                                                      index_name=index_name,
                                                      date_find=date_find, fund=fund, index=index_position,
                                                      position=position, output=output, industry_flag=True)
            output.csv_save(data=MSE, index1 = (datelist[1:]+['ave']), column = (fundlist+['ave']), path='MSE_result.xlsx')
            output.csv_save(data=R1, index1 = (datelist[1:]+['ave']), column = (fundlist+['ave']), path='R1_result.xlsx')
            output.csv_save(data=adjustR1, index1 = (datelist[1:]+['ave']), column = (fundlist+['ave']), path='adjustR1_result.xlsx')
            output.csv_sheet_save(data=res_final, path='predict_result.xlsx', index_name=index_name, fund_name=fundlist)
            fund_type_final = pd.DataFrame(fund_type, columns=['日期', '基金', '类型'])
            fund_type_final.to_excel(self.result_dir + '/基金类型预测结果.xlsx')

        elif model_type == 'OLS':#R2, adjustR2,
            res_final, MSE, R1, adjustR1,  fund_type = self.linear_model(datelist = datelist[1:], fundlist = fundlist, fund_name = fund_name,
                                                                           index_name = index_name, date_find = date_find,
                                                                           fund = fund, index = index, position=position, each_len=each_len,
                                                                           output=output, lamda=0, lamda_flag=False,
                                                                           PCA_n = 0, PCA_flag=False)
            output.csv_save(data = MSE, index1 =(datelist[1:]+['ave']), column = (fundlist+['ave']), path = 'MSE_result.xlsx')
            output.csv_save(data = R1, index1 = (datelist[1:]+['ave']), column = (fundlist+['ave']), path = 'R1_result.xlsx')
            output.csv_save(data = adjustR1, index1 = (datelist[1:]+['ave']), column = (fundlist+['ave']), path = 'adjustR1_result.xlsx')
            #output.csv_save(data = R2, index1 = (list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path = 'R2_result.xlsx')
            #output.csv_save(data = adjustR2, index1 = (list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path = 'adjustR2_result.xlsx')
            output.csv_sheet_save(data = res_final, path = 'predict_result.xlsx', index_name = index_name, fund_name = fundlist)
            fund_type_final = pd.DataFrame(fund_type, columns=['日期', '基金', '类型'])
            fund_type_final.to_excel(self.result_dir + '/基金类型预测结果.xlsx')

        elif model_type == 'OLS+PCA':#R2,adjustR2,
            res_final, MSE, R1,  adjustR1,  fund_type = self.linear_model(datelist=datelist[1:], fundlist=fundlist, fund_name=fund_name,
                                                                           index_name=index_name, date_find = date_find,
                                                                           fund=fund, index=index, position=position,
                                                                           each_len=each_len, output=output, lamda=0, lamda_flag=False,
                                                                           PCA_n=PCA_n, PCA_flag=True)
            output.csv_save(data=MSE, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='MSE_result.xlsx')
            output.csv_save(data=R1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='R1_result.xlsx')
            output.csv_save(data=adjustR1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='adjustR1_result.xlsx')
            #output.csv_save(data=R2, index1=(list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path='R2_result.xlsx')
            #output.csv_save(data=adjustR2, index1=(list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path='adjustR2_result.xlsx')
            output.csv_sheet_save(data=res_final, path='predict_result.xlsx', index_name=index_name, fund_name=fundlist)
            fund_type_final = pd.DataFrame(fund_type, columns=['日期', '基金', '类型'])
            fund_type_final.to_excel(self.result_dir + '/基金类型预测结果.xlsx')

        elif model_type == 'ridge':#R2, adjustR2,
            res_final, MSE, R1, adjustR1,  fund_type = self.linear_model(datelist=datelist[1:], fundlist=fundlist, fund_name=fund_name,
                                                                           index_name=index_name, date_find=date_find,
                                                                           fund=fund, index=index, position=position,
                                                                           each_len=each_len, output=output, lamda=lamda,
                                                                           lamda_flag=True, PCA_n=PCA_n, PCA_flag=False)
            output.csv_save(data=MSE, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='MSE_result.xlsx')
            output.csv_save(data=R1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='R1_result.xlsx')
            output.csv_save(data=adjustR1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='adjustR1_result.xlsx')
            #output.csv_save(data=R2, index1=(list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path='R2_result.xlsx')
            #output.csv_save(data=adjustR2, index1=(list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path='adjustR2_result.xlsx')
            output.csv_sheet_save(data=res_final, path='predict_result.xlsx', index_name=index_name, fund_name=fundlist)
            fund_type_final = pd.DataFrame(fund_type, columns=['日期', '基金', '类型'])
            fund_type_final.to_excel(self.result_dir + '/基金类型预测结果.xlsx')

        else:#adjustR2, R2,
            res_final, MSE, R1, adjustR1, fund_type = self.linear_model(datelist=datelist[1:], fundlist=fundlist, fund_name=fund_name,
                                                                           index_name=index_name, date_find=date_find,
                                                                           fund=fund, index=index, position=position, each_len=each_len,
                                                                           output=output, lamda=lamda, lamda_flag=True, PCA_n=PCA_n, PCA_flag=True)
            output.csv_save(data=MSE, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='MSE_result.xlsx')
            output.csv_save(data=R1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='R1_result.xlsx')
            output.csv_save(data=adjustR1, index1=(datelist[1:]+['ave']), column = (fundlist+['ave']), path='adjustR1_result.xlsx')
            #output.csv_save(data=R2, index1=(list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path='R2_result.xlsx')
            #output.csv_save(data=adjustR2, index1=(list(fund.index)[each_len-1:]+['ave']), column = (fundlist+['ave']), path='adjustR2_result.xlsx')
            output.csv_sheet_save(data=res_final, path='predict_result.xlsx', index_name=index_name, fund_name=fundlist)
            fund_type_final = pd.DataFrame(fund_type, columns=['日期', '基金', '类型'])
            fund_type_final.to_excel(self.result_dir + '/基金类型预测结果.xlsx')

        return

'''if __name__=='__main__':
    Algo_fund_position(result_dir='./result').main(model_type = 'reg', datelist = ['20210630', '20210930', '20211231', '20220331'], fundlist = ['华宝生态中国'], each_len = 24, PCA_n = 0.9, lamda = 0.001)
'''
''''''