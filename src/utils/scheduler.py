# -*- encoding: utf-8 -*-
"""
@File    :   scheduler.py
@Time    :   2025/05/02 12:11:54
@Author  :   yxing
"""

import numpy as np

class Scheduler():
    def __init__(self, start, end, mode='linear', **kwargs):
        
        self.mode = mode
        self.idx = 0

        if self.mode == 'linear':
            assert 'n_step' in kwargs, "key word parameter ''n_step'' missing"
            self.scheduler = self.linear_scheduler(
                start, end, kwargs['n_step']
            )
        elif self.mode == 'warmup':
            self.scheduler = self.warmup_scheduler(
                start, end, kwargs['t'], kwargs['t_total']
            )
        else:
            raise NotImplementedError("No implementation for {} scheduler mode.".format(self.mode))

    def next(self):
        v = self.scheduler[self.idx]
        self.idx+=1
        return v

    def linear_scheduler(self, start, end, n_step):
        if start > end:
            interval = (start - end) / n_step
            return list(reversed(np.arange(end, start+1e-6, interval)))
        else:
            interval = (end - start) / n_step
            return list(np.arange(start, end+1e-6, interval))

    def warmup_scheduler(self, start, end, t, t_total):
        return min(end, start + (end - start) * (t / t_total))