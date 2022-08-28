# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Output():
    '''
    输出
    '''
    def __init__(self, result_dir):
        '''
        :param result_dir str: 输出文件夹
        '''
        self.result_dir = result_dir

    def plot(self, title, saveFigPath, res1, res2, index_name):
        '''
        画图并保存

        :param title str: 图表标题
        :param savefigpath str: 保存图片的路径
        :param res1 list: 预测的仓位
        :param res2 list: 实际仓位
        :param index_name list: 行业名称
        '''

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        index_name = [i[:-4] for i in index_name]
        x = np.arange(len(index_name))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(30,20))
        rects1 = ax.bar(x - width / 2, res1, width, label='predict')
        rects2 = ax.bar(x + width / 2, res2, width, label='real')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.tick_params(axis="y", which="major", labelsize=15)
        ax.set_ylabel('Position', fontsize=15)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(index_name, fontsize=15)
        ax.legend()

        fig.tight_layout()
        plt.savefig(self.result_dir + '/' + saveFigPath)
        plt.show()

    def csv_save(self, data, index1, column, path):
        '''
        将数据写入Excel

        :param data list: 写入数据
        :param index1 list: data的行名
        :param path str：写入Excel的文件名
        '''
        #print(index1)
        #print(column)
        data = pd.DataFrame(data, index = index1, columns=column)
        data.to_excel(self.result_dir + '/' + path)

    def csv_sheet_save(self, data, path, index_name, fund_name):
        '''
        将数据分sheet写入Excel

        :param data dict: 写入数据
        :param path str：写入Excel的文件名
        :param fund_name list：所有基金的名字列表
        :param index_name list：行业的名字列表
        '''

        writer = pd.ExcelWriter(self.result_dir + '/' + path)
        #print(fund_name)
        #print(len(fund_name))
        for k, v in data.items():
            results = pd.DataFrame(v, index=fund_name, columns=index_name)
            results.to_excel(excel_writer=writer, sheet_name=k)
        writer.save()
        writer.close()