# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

# --------------------------------------------------------------
# 2. Main문
    # 12) Train & Update Logger & Update Checkpoint
        # (3) Save Values (Close Logger / Close Writer / Save Checkpoint)
# --------------------------------------------------------------
# 1] Save Image from Plot
def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

# --------------------------------------------------------------
# 2. Main문
    # 8) Logger 설정 + Resume Checkpoint
        # (2) Resume 설정하지 않은 경우 : Start Checkpoint
            # 1] Logger 열기
# --------------------------------------------------------------
class Logger(object):
    '''Save training process to log file with simple plot function.'''
    # [1] 초기 설정
    def __init__(self, fpath, title=None, resume=False):
        # 1]] 변수 선언
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        # 2]] 파일이 존재하는 경우
        if fpath is not None:
            # [1] Resume인 경우 : Load File (Append Mode)
            if resume:
                # 1]] Read Names in Previous File -> Write Names in New File
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []
                # 2]] Read Numbers in Previous File -> Write Numbers in New File
                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                # 3]] Load File (Append Mode)
                self.file = open(fpath, 'a')
            # [2] Resume 아닌 경우 : Load File (Writing Mode)
            else:
                self.file = open(fpath, 'w')

    # [2] Write Name in File
    def set_names(self, names):
        # 1]] resume인 경우 : Pass
        if self.resume: 
            pass
        # 2]] 변수 선언 (Numbers / Names)
        self.numbers = {}
        self.names = names
        # 3]] Write Name in New File
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    # [3] Append Numbers in File
    def append(self, numbers):
        # 1]] Name 개수 고정
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        # 2]] Append Values in New File
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    # [4] Append Numbers in File
    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    # [5] Close Logger
    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
                    
if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
    'resadvnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt', 
    'resadvnet32':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
    'resadvnet44':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
    }

    field = ['Valid Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('test.eps')
