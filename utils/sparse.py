from utils.prune_utils import *
from utils.opt import opt


class BNSparse():
    def __init__(self,bn_txt):
        self.prune_idx =[]
        self.bn_file = bn_txt

    def build_sparse(self, model):
        if opt.prune == 1:
            CBL_idx, _, self.prune_idx, shortcut_idx, _ = parse_module_defs2(model.module_defs)
            if opt.sr:
                print('shortcut sparse training')
                print('pruned idx:', self.prune_idx)
        elif opt.prune == 0:
            CBL_idx, _, self.prune_idx = parse_module_defs(model.module_defs)
            if opt.sr:
                print('normal sparse training ')

    def update_BN(self, model):
        for idx in self.prune_idx:
            bn_module = model.module_list[idx][1]
            bn_module.weight.grad.data.add_(opt.s * torch.sign(bn_module.weight.data))  # L1

    def write_tensorboard(self, model, tb_writer):
        for idx in self.prune_idx:
            bn_weights = gather_bn_weights(model.module_list, [idx])
            tb_writer.add_histogram('before_train_perlayer_bn_weights/hist', bn_weights.numpy(), idx, bins='doane')

    def write_tensorboard_after(self, model, tb_writer):
        for idx in self.prune_idx:
            bn_weights = gather_bn_weights(model.module_list, [idx])
            tb_writer.add_histogram('after_train_perlayer_bn_weights/hist', bn_weights.numpy(), idx, bins='doane')

    def write_txt(self, model, tb_writer, epoch):
        bn_weights = gather_bn_weights(model.module_list, self.prune_idx)
        bn_numpy = bn_weights.numpy()
        with open(self.bn_file, "a+") as f:
            f.write("Epoch--->{}, bn_ave--->{}, bn_var--->{}\n".
                    format(epoch, str(np.mean(bn_numpy)), str(np.var(bn_numpy))))
        tb_writer.add_histogram('bn_weights/hist', bn_numpy, epoch, bins='doane')

    def draw_bn(self, model,):
        bn_weights = gather_bn_weights(model.module_list, self.prune_idx)
        bn_numpy = bn_weights.numpy()
        return np.mean(bn_numpy)