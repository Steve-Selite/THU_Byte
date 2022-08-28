# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict

class DataProcess():
    '''
    数据处理
    '''

    def __init__(self, data_dir):
        '''
        param: data_dir string: 文件路径，注意文件要是csv格式或者xlsx格式
        '''
        self.data_dir = data_dir

    def revise(self, df, col):
        '''
        提取dataframe的名字并截取相应的行和列

        :param: df dataframe: 处理的数据框
        :param: col int: 截取的起始列
        :return: df_name list: 数据框的名字
        :return: df dataframe：处理完的数据框
        '''

        df_name = list(df.columns)[col:]
        df_columns = list(df.iloc[0, col:])
        df_index = list(df.iloc[:, 0])[1:]
        df = df.iloc[1:, col:]
        df.index = df_index
        df.columns = df_columns
        return df_name, df

    def industry(self, data_industry, fundlist, datelist, index_name, fund_code, fund_name):
        '''
        根据公告的股票仓位获得基金的实际行业仓位

        :param: data_industry str: 行业Excel表
        :param datelist list：日期列表
        :param fundlist list：所预测基金的名字列表
        :param index_name list：行业的名字列表
        :param fund_code list：所有基金的代码
        :param fund_name list：所有基金的名字列表
        :return: position dict：key为日期，value为二维list，对应每个基金在该日的实际仓位
        '''

        indu = pd.read_excel(data_industry)
        position = defaultdict(list)
        index_name_new = []
        for name in index_name:
            index_name_new.append(name[:-4])
        for date in datelist:
            posit = []
            for name in fundlist:
                pos = pd.read_excel(self.data_dir + '/' + date + '-position.xlsx',
                                    sheet_name=fund_code[fund_name.index(name)])
                pos["股票代码"] = pos["symbol"].map(lambda x: int(x[:-3]), na_action='ignore')
                pos1 = pd.concat([pos['mkv'], pos['股票代码']], axis=1)
                p = pos1.set_index('股票代码').join(indu.set_index('股票代码'), on="股票代码", how="left")
                l = p.groupby(by=['行业名称'])['mkv'].sum()
                new_name = list(l.index)
                l1 = np.zeros(len(index_name))
                for i in range(len(index_name_new)):
                    n = index_name_new[i]
                    if n in new_name:
                        l1[i] = l[n]
                l1 = l1 / sum(l1)
                posit.append(l1)
            position[date] = posit
        return position

    def industry_index(self, data_industry, fundlist, datelist, index_name, fund_code, fund_name, date_find, fund):
        '''
        根据公告的股票仓位获得基金的实际行业仓位

        :param: data_industry str: 行业Excel表
        :param datelist list：日期列表
        :param fundlist list：所预测基金的名字列表
        :param index_name list：行业的名字列表
        :param fund_code list：所有基金的代码
        :param fund_name list：所有基金的名字列表
        :param date_find list：每季度最后一个交易日在datelist中对应的位置
        :param fund dataframe：index为时间，columns为基金code的基金收益率数据
        :return: position dict：key为日期，value为二维list，对应每个基金在该日的实际仓位
        '''

        indu = pd.read_excel(data_industry)
        position = defaultdict(pd.DataFrame)
        index_name_new = []
        for name in index_name:
            index_name_new.append(name[:-4])
        for date in datelist[:-1]:
            pos_date = datelist.index(date)
            for name in fundlist:
                pos = pd.read_excel(self.data_dir + '/' + date + '-position.xlsx',
                                    sheet_name=fund_code[fund_name.index(name)])
                pchg = pd.read_excel(self.data_dir + '/' + date + '-position-pchg.xlsx', sheet_name=fund_code[fund_name.index(name)])
                trade_date = list(pchg.iloc[:,0])
                #print(trade_date)
                pchg = pchg.iloc[:,1:]
                #print(pchg)
                pchg.index = trade_date
                pchg.columns = pchg.columns.map(lambda x: int(x[:-3]), na_action='ignore')
                #print(pchg)
                pchg = pchg.pct_change().iloc[1:, :]  #有问题
                #print(pchg)
                #pchg = pchg.set_index('trade_date')


                #print(pchg)
                pos["股票代码"] = pos["symbol"].map(lambda x: int(x[:-3]), na_action='ignore')
                pos1 = pd.concat([pos['mkv'], pos['股票代码']], axis=1)
                p = pos1.set_index('股票代码').join(indu.set_index('股票代码'), on="股票代码", how="left")
                l = p.groupby(by=['行业名称'])['mkv'].sum()
                #print(l)
                #print(p)
                sum_mkv = []
                for i in p.index:
                    if pd.isnull(p.loc[i, '行业名称']):
                        sum_mkv.append(1)
                    else:
                        sum_mkv.append(l[p.loc[i, '行业名称']])
                p['mkv_sum'] = sum_mkv
                p['mkv_pro'] = p['mkv'] / p['mkv_sum']
                p_final = pd.DataFrame(np.array(p['mkv_pro']).reshape(1, -1), index=[date], columns=list(p.index))
                #print(p_final)
                datelist_this = list(pchg.index)
                p_final = pd.DataFrame(np.repeat(p_final.values, len(datelist_this), axis=0), index=datelist_this, columns=p_final.columns)
                #print(p_final)
                pchg1 = p_final * pchg
                column_index = defaultdict(list)
                #print(pchg1)
                for i in index_name_new:
                    column_index[i] = []
                for i in range(len(p.index)):
                    name_indu = p.loc[p.index[i], '行业名称']
                    if not pd.isnull(name_indu):
                        column_index[name_indu].append(i)
                for i in index_name_new:
                    if column_index[i] == []:
                        pchg1[i] = list(fund.iloc[date_find[pos_date]:date_find[pos_date+1], index_name_new.index(i)].values)
                    else:
                        for j in range(len(column_index[i])):
                            if j == 0:
                                pchg1[i] = pchg1.iloc[:, column_index[i][j]]
                            else:
                                pchg1[i] += pchg1.iloc[:, column_index[i][j]]
                #print(pchg1)
                position[name] = pd.concat([position[name],pchg1.loc[:,index_name_new]])
                #print(position)
        #print(position)
        return position

    def time_find(self, datelist, index):
        '''
        在完整数据表中找每个季度最后一个交易日的时间位置

        :param datelist list：日期列表
        :param index list：index为时间，columns为基金code的行业收益率数据
        :return: time_pos list：每季度最后一个交易日对应在datelist中的位置
        '''

        date_str = []
        for date in index:
            strtime = date.strftime("%Y-%m-%d %H:%M:%S")[:10].split('-')
            strtime1 = strtime[0] + strtime[1] + strtime[2]
            date_str.append(strtime1)
        time_pos = [0]
        for date in datelist:
            if date in date_str:
                time_pos.append(date_str.index(date)+1)
            else:
                time_pos.append(len(index))
        return time_pos

    def main(self, data_fund, col, data_industry, datelist, fundlist):
        '''
        主函数

        :param data_fund list：data_fund[0]为基金数据所在文件名，data_fund[1]为基金收益率的sheet名称，data_fund[2]为指数收益率的sheet名称
        :param col list：col[0]为基金的列数限制，col[1]为指数的列数限制
        :param: data_industry str: 行业Excel表
        :param datelist list：日期列表
        :param fundlist list：所预测基金的名字列表
        :return fund dataframe：index为时间，columns为基金code的基金收益率数据
        :return index list：index为时间，columns为基金code的行业收益率数据
        :return position dict：key为日期，value为二维list，对应每个基金在该日的实际仓位
        :return date_find list：每季度最后一个交易日在datelist中对应的位置
        :return fund_name list：所有基金的名字列表
        :return index_name list：行业的名字列表
        :return index_position list：行业的名字列表

        '''

        fund = pd.read_excel(self.data_dir+data_fund[0], sheet_name=data_fund[1])
        index = pd.read_excel(self.data_dir+data_fund[0], sheet_name=data_fund[2])
        fund_name, fund = self.revise(df=fund, col=col[0])
        index_name, index = self.revise(df=index, col=col[1])
        print(index_name)
        position = self.industry(data_industry=self.data_dir+data_industry, fundlist=fundlist, datelist=datelist, index_name=index_name,
                                 fund_code=fund.columns, fund_name=fund_name)
        fund = fund.pct_change().iloc[1:, :]
        index = index.pct_change().iloc[1:, :]
        date_find = self.time_find(datelist=datelist[1:], index=list(fund.index))
        index_position = self.industry_index(data_industry=self.data_dir + data_industry, fundlist=fundlist,
                                             datelist=datelist, index_name=index_name,
                                             fund_code=fund.columns, fund_name=fund_name, date_find=date_find,
                                             fund=index)
        return fund, index, position, date_find, fund_name, index_name, index_position

'''if __name__=='__main__':
    fund, index, position, date_find, fund_name, index_name, index_position= DataProcess(data_dir='./data').main(data_fund=['/data-fund.xlsx','nav_adj公募基金复权单位净值', 'sector_index板块和申万一级行业'], col=[1,6], data_industry = '/行业.xlsx', datelist = ['20210331','20210630', '20210930', '20211231', '20220331'], fundlist = ['景顺长城策略精选','万家汽车新趋势A','万家成长优选A','华安动态灵活配置A','万家臻选','中金新锐A',
                                                            '北信瑞丰健康生活主题','华宝资源优选A','华商新锐产业','国投瑞银策略精选','华宝国策导向',
                                                            '万家新兴蓝筹','建信中国制造2025A','建信健康民生A','信达澳银核心科技','前海开源股息率100强','华宝收益增长',
                                                            '中泰玉衡价值优选','景顺长城成长之星','国泰鑫睿','工银瑞信新机遇A','海富通改革驱动',
                                                            '华泰柏瑞量化先行A','建信弘利','国泰聚优价值A','华夏新锦绣A','南方中国梦',
                                                            '融通内需驱动AB','浦银安盛新经济结构A','建信改革红利','诺安安鑫','中信保诚精萃成长',
                                                            '信达澳银精华A','华宝宝康灵活','大成灵活配置','东吴新趋势价值线','平安新鑫先锋A','海富通收益增长',
                                                            '中银主题策略A','广发睿毅领先A','中泰星元价值优选A','东方主题精选','鹏华价值精选',
                                                            '金鹰中小盘精选','华商科技创新'])
    print(fund)
    print(index)
    print(position)
    print(date_find)
    print(index_position)'''