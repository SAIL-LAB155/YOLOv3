from utils.opt import opt
import torch
from cfg import config
from utils.pytorchtools import EarlyStopping
from torch import optim
class Optimizer():
    def build(self, model):
        pg0, pg1 = [], []  # optimizer parameter groups
        for k, v in dict(model.named_parameters()).items():
            if 'Conv2d.weight' in k:
                pg1 += [v]  # parameter group 1 (apply weight_decay)
            else:
                pg0 += [v]  # parameter group 0

        if opt.optimize == 'adam':  # 将网络数数放到优化器
            optimizer = torch.optim.Adam(pg0, lr=config.hyp['lr0'])
            # optimizer = AdaBound(pg0, lr=config.hyp['lr0'], final_lr=0.1)
        elif opt.optimize == 'sgd':
            optimizer = torch.optim.SGD(pg0, lr=config.hyp['lr0'], momentum=config.hyp['momentum'], nesterov=True)
        else:
            raise Exception
        optimizer.add_param_group(
            {'params': pg1, 'weight_decay': config.hyp['weight_decay']})  # add pg1 with weight_decay
        del pg0, pg1
        return optimizer


class LR_Scheduler:
    def __init__(self):
        self.patience = 7
        self.warm_up = 6
        self.patience_decay = {1: 0.8, 2: 0.5, 3: 0}
        self.early_stoping = EarlyStopping(patience=config.patience, verbose=True)
        self.decay = 0
        self.stop = False
        self.lr_decay_dict = [0.7,0.9]

    def init_scheduler(self):
        schedule_cfg = config.lr_schedule.values()
        name = config.lr_schedule['name']
        Scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = Scheduler(optimizer=self.optimizer, **schedule_cfg)

    def update(self, epoch, optimizer, iteration, epoch_size):
        if epoch < self.warm_up:
            lr = self.warmup_schl(iteration, epoch_size)
        else:
            lr = self.normal_scl()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_schl(self,optimizer, iteration, epoch_size):
        self.lr = 1e-6 + (config.hyp['lr0'] - 1e-6) * iteration / (epoch_size * config.warm_up)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr, optimizer

    def lr_decay(self, optimizer,epoch):
        epoch_rate = epoch / opt.epochs
        if epoch_rate < self.lr_decay_dict[0]:
            self.lr = opt.LR
        elif self.lr_decay_dict[0] < epoch_rate < self.lr_decay_dict[1]:
            self.lr = self.lr * 0.1
        else:
            self.lr = self.lr * 0.1
        for pg in optimizer.param_groups:
            pg["lr"] = self.lr
        return optimizer, self.lr

    def normal_scl(self, epoch, optimizer,scheduler):
        if epoch == config.warm_up:
            self.lr = opt.LR
        if epoch > config.warm_up:
            optimizer, self.lr = self.lr_decay(optimizer,epoch)
            # optimizer, self.lr = self.cosin_lr(optimizer,scheduler)
        elif epoch < config.warm_up:
            return optimizer, self.lr
        return optimizer, self.lr

    def draw_lr(self, index, value):
        pass

    def cosin_lr(self, optimizer,scheduler,):
        # pass
        scheduler.step()
        return optimizer, self.lr
    def step_lr(self):
        pass

def plot_feature():
# yolo_feature = model.yolofeature.cpu().numpy()
# model.yolofeature.zero_()
# yolo_feature = yolo_feature[0, :, :, :]
# size = yolo_feature.shape[0]
# for i in range(size):  # 可视化了32通道
#     ax = plt.subplot(2, size/2, i + 1)
#     ax.set_title('Feature {}'.format(i))
#     ax.axis('off')
#     ax.set_title('new—conv1-image')
#     plt.imshow(yolo_feature[i, :, :], cmap='jet')
# plt.show()
    pass