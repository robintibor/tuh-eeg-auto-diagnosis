import math


class CosineWithWarmRestarts(object):
    def __init__(self, optimizer, batch_period, base_lr, base_wd = 0, update_wd = 0):

        self.optimizer = optimizer
        self.base_lr = base_lr
        self.base_wd = base_wd
        self.batch_iteration = 0
        self.batch_period = batch_period
        self.update_wd = update_wd
        self.m_mult = 1.0
        self.t_mult = 1.0

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            self.batch_iteration = self.batch_iteration = self.batch_iteration + 1
        else:
            self.batch_iteration = batch_iteration
        tt = self.batch_iteration / self.batch_period
        multiplier = 0.5 * (1.0 + math.cos(tt * math.pi))
        cur_lr = self.base_lr * multiplier
        cur_wd = self.base_wd * multiplier
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = cur_lr
            if (self.update_wd == 1):    param_group['weight_decay'] = cur_wd
        return cur_lr


class ScheduledOptimizer(object):
    def __init__(self, scheduler):
        assert hasattr(scheduler, 'optimizer')
        self.scheduler = scheduler

    def state_dict(self):
        self.scheduler.optimizer.state_dict()

    def step(self):
        self.scheduler.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.scheduler.optimizer.zero_grad()