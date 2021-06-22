import csv
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import test  # import test.py to get mAP after each epoch
from models import *
from utils.compute_flops import print_para_time_flops
from utils.datasets import *
from utils.opt import opt
from utils.prune_utils import *
from utils.sparse import BNSparse
from utils.train_utils import Optimizer, LR_Scheduler
from utils.utils import *

mixed_precision = config.mixed_precision
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Trainer:
    def __init__(self):
        self.weights = opt.weights
        if opt.finetune:
            self.cfg = opt.cfg
        else:
            self.cfg = os.path.join('cfg', 'yolov3-' + opt.type + '-' + '1cls' + '-' + opt.activation + '.cfg')
        self.data = opt.data
        self.img_size = opt.img_size
        self.epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
        self.batch_size = opt.batch_size
        self.accumulate = opt.accumulate
        self.multi_scale = opt.multi_scale
        self.cur_epoch = 0
        self.sparse = opt.sr
        self.best_result = [float("inf"), float("inf"), 0, float("inf"), 0, 0, 0, 0, float("inf"), float("inf"), 0,float("inf")]
        self.train_loss_ls, self.val_loss_ls, self.prmf_ls = [], [], []
        self.best_epoch = 0
        self.fitness = 0
        self.tb_writer = SummaryWriter('tensorboard/{}/{}'.format(opt.expFolder, opt.expID))
        self.build_dir()
        self.write_txt_title()
        self.cal_img_size()
        self.Optimizer = Optimizer()
        self.LR_Scheduler = LR_Scheduler()
        if self.sparse:
            self.BNsp = BNSparse(self.bn_file)


    def cal_img_size(self):
        if self.multi_scale:
            self.img_sz_min = round(self.img_size / 32 / 1.5) + 1
            self.img_sz_max = round(self.img_size / 32 * 1.5) - 1
            self.img_size = self.img_sz_max * 32  # initiate with maximum multi_scale size
            print('Using multi-scale %g - %g' % (self.img_sz_min * 32, self.img_size))

    def init_scheduler(self):
        if opt.lr_schedule == 'cosin':
            print('Using cosin lr schedule')
            lr_schedule = {'name': 'CosineAnnealingLR', 'T_max': opt.epochs, 'eta_min': 0.00001}
        elif opt.lr_schedule == 'step':
            print('Using step lr schedule')
            lr_schedule = {'name': 'MultiStepLR', 'milestones': [0.7*opt.epochs,0.9*opt.epochs], 'gamma': 0.1}
        # schedule_cfg = lr_schedule
        name = lr_schedule.pop('name')
        Scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = Scheduler(optimizer=self.optimizer, **lr_schedule)

    def build_dir(self):
        # i.e ./gray/spp/1(./opt.expFolder/opt.type/opt.expID)
        self.result_dir = os.path.join(opt.expFolder, opt.expID)
        self.train_dir = os.path.join('result', self.result_dir) + os.sep  # train result dir
        if opt.finetune:
            self.weight_dir = os.path.join('finetune', opt.expFolder) + os.sep  # weights dir
        else:
            self.weight_dir = os.path.join('weights', self.result_dir) + os.sep  # weights dir
        self.bn_file = os.path.join(self.train_dir, 'bn.txt')
        if "last" not in self.weights or not opt.resume:
            if 'test' not in self.weight_dir or not os.path.exists(self.weight_dir):
                os.makedirs(self.weight_dir)
        self.last = self.weight_dir + 'last.pt'
        self.best = self.weight_dir + 'best.pt'
        os.makedirs(self.train_dir, exist_ok=True)
        self.results_txt = os.path.join(self.train_dir, 'results.txt')  # results.txt in weights_folder
        self.weights = self.last if opt.resume else opt.weights  # if resume use the last
        data_dict = parse_data_cfg(self.data)
        self.train_path = data_dict['train']
        self.nc = int(data_dict['classes'])  # number of classes

    def build_model(self):
        self.model = Darknet(self.cfg, (self.img_size, self.img_size), arc=opt.arc).to(device)
        self.cutoff = -1  # backbone reaches to cutoff layer
        self.start_epoch = 0
        self.best_fitness = 0.
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            chkpt = torch.load(self.weights, map_location=device)

            # load model
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if
                              self.model.state_dict()[k].numel() == v.numel()}
            self.model.load_state_dict(chkpt['model'], strict=False)
            print('loaded weights from', self.weights, '\n')

            # load optimizer
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(self.results_txt, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt
            self.start_epoch = chkpt['epoch'] + 1
            del chkpt

        elif len(self.weights) > 0:  # darknet format
            # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            self.cutoff = load_darknet_weights(self.model, self.weights)
            print('loaded weights from', self.weights, '\n')

        if opt.freeze and opt.type in ['spp', 'original']:
            # spp , original
            for k, p in self.model.named_parameters():
                # if 'BatchNorm2d' in k and int(k.split('.')[1]) > 33: #open bn
                if 'BatchNorm2d' in k:
                    p.requires_grad = False
                elif int(k.split('.')[1]) < 33:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

        elif opt.freeze and opt.type == 'tiny':
            # tiny
            for k, p in self.model.named_parameters():
                # if 'BatchNorm2d' in k and int(k.split('.')[1]) > 33: #open bn
                if 'BatchNorm2d' in k:
                    p.requires_grad = False
                elif int(k.split('.')[1]) < 9:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

        self.model.nc = self.nc  # attach number of classes to model
        self.model.arc = opt.arc  # attach yolo architecture
        self.model.hyp = config.hyp  # attach hyperparameters to model
        torch_utils.model_info(self.model, report='summary')  # 'full' or 'summary'
        # Initialize distributed training
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0, 1, 2])
            self.model.module_list = self.model.module.module_list
            self.model.yolo_layers = self.model.module.yolo_layers  # move yolo layer indices to top level
        self.flops, self.params, self.infer_time = print_para_time_flops(self.model)

    def build_dataloader(self):
        self.dataset = LoadImagesAndLabels(self.train_path,
                                      self.img_size,
                                      self.batch_size,
                                      augment=True,
                                      hyp=config.hyp,  # augmentation hyperparameters
                                      rect=opt.rect,  # rectangular training
                                      image_weights=False,
                                      cache_labels=True if self.epochs > 10 else False,
                                      cache_images=False)

        # Dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=2,
                                                 shuffle=not opt.rect,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=self.dataset.collate_fn)

    def update_map(self, results):
        P, self.fitness= results[0], results[2]  # mAP
        if self.fitness > self.best_fitness and P > 0.5:
            self.best_fitness = self.fitness
            self.best_epoch = self.cur_epoch

    def write_txt_title(self):
        with open(self.results_txt, 'a') as file:
            file.write(('%10s' * 18) % (
                'Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'soft', 'rratio', 'targets', 'img_size', 'lr',
                "P", "R", "mAP", "F1", "test_GIoU", "test_obj", "test_cls\n"))

    def write_txt_result(self, results, s):
        with open(self.results_txt, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

    def write_csv(self, results, *mloss):
        exist = os.path.exists(self.train_dir + 'train_csv.csv')
        with open(self.train_dir + 'train_csv.csv', 'a+', newline='')as f:
            f_csv = csv.writer(f)
            if not exist:
                title = ['Epoch', 'GIoU', 'obj', 'cls', 'total',
                         'lr', "P", "R", "mAP", "F1", "test_GIoU", "test_obj", "test_cls"]
                f_csv.writerow(title)
            info_str = [self.cur_epoch, [x.cpu().tolist() for x in mloss], self.lr]
            info_str.extend(list(results))
            f_csv.writerow(info_str)

            self.train_loss_ls.append([x.cpu().tolist() for x in mloss])
            self.val_loss_ls.append(list(results)[-3:])
            self.prmf_ls.append(list(results)[0:4])

    def write_tensorboard(self, results, msoft_target, *mloss):
        if self.tb_writer:
            x = list(mloss) + list(results) + [msoft_target]
            for xi, title in zip(x, config.titles):
                self.tb_writer.add_scalar(title, xi, self.cur_epoch)
            self.tb_writer.add_scalar('lr', self.lr, self.cur_epoch)
            self.best_result = update_result(self.best_result, x)


    def write_whole_csv(self, train_time, final_epoch):
        whole_result = os.path.join('result', opt.expFolder, "{}_result_{}.csv".format(opt.expFolder, config.computer))
        exist = os.path.exists(whole_result)
        with open(whole_result, "a+") as f:
            f_csv = csv.writer(f)
            if not exist:
                title = [
                    'ID', 'tpye', 'activation', 'batch_size', 'optimize', 'freeze', 'epoch_num', 'LR', 'weights',
                    'multi-scale', 'img_size', 'rect', 'data', 'model_location', 'folder_name', 'parameter',
                    'flops', 'infer_time', 'train_GIoU', 'train_obj', 'train_cls', 'total', "P", "R", "mAP", "F1",
                    "val_GIoU", "val_obj", "val_cls", 'train_time', 'final_epoch', 'best_epoch',
                ]
                f_csv.writerow(title)
            infostr = [
                opt.expID, opt.type, opt.activation, opt.batch_size, opt.optimize, opt.freeze, opt.epochs,
                opt.LR, opt.weights, self.multi_scale, opt.img_size, opt.rect, opt.data, config.computer,
                self.train_dir, self.params, self.flops, self.infer_time
            ]
            self.best_result = res2list(self.best_result)
            infostr.extend(self.best_result[:-1])
            infostr.extend([train_time, final_epoch, self.best_epoch])
            f_csv.writerow(infostr)

    def save_model(self, final_epoch):
        with open(self.results_txt, 'r') as f:
            # Create checkpoint
            chkpt = {'epoch': self.cur_epoch,
                     'best_fitness': self.best_fitness,
                     'training_results': f.read(),
                     'model': self.model.module.state_dict() if type(
                         self.model) is nn.parallel.DistributedDataParallel else self.model.state_dict(),
                     'optimizer': None if final_epoch else self.optimizer.state_dict()}

        # Save last checkpoint
        torch.save(chkpt, self.last)
        if config.convert_weight:
            convert(cfg=self.cfg, weights=self.weight_dir + 'last.pt')
            os.remove(self.weight_dir + 'last.pt')

        # Save best checkpoint
        if self.best_fitness == self.fitness:
            torch.save(chkpt, self.best)
            if config.convert_weight:
                convert(cfg=self.cfg, weights=self.weight_dir + 'best.pt')
                os.remove(self.weight_dir + 'best.pt')

        # Save backup every 10 epochs (optional)
        if self.cur_epoch > config.start_save_epoch and self.cur_epoch % opt.save_interval == 0:
            torch.save(chkpt, self.weight_dir + 'backup%g.pt' % self.cur_epoch)
            if config.convert_weight:
                convert(cfg=self.cfg, weights=self.weight_dir + 'backup%g.pt' % self.cur_epoch)
                os.remove(self.weight_dir + 'backup%g.pt' % self.cur_epoch)
        # Delete checkpoint
        del chkpt

    def train(self):
        init_seeds()
        # Remove previous results
        for f in glob.glob('*_batch*.jpg') + glob.glob(self.results_txt):
            os.remove(f)

        # Initialize model
        self.build_model()
        if self.sparse:
            self.BNsp.build_sparse(self.model)
        # Optimizer
        self.optimizer = self.Optimizer.build(self.model)
        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            print('start mixed precision training')
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=1)
        self.init_scheduler()
        # Dataset and dataloader
        self.build_dataloader()
        #write bn tensorboard
        if self.sparse:
            self.BNsp.write_tensorboard(self.model, self.tb_writer)
        # Start training
        nb = len(self.dataloader)
        print('nb',nb)
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        t0 = time.time()
        print('Starting %s for %g epochs...' % ('training', self.epochs))
        final_epoch = 0
        x, y, b_weight= [], [], []
        # epoch ------------------------------------------------------------------
        for self.cur_epoch in range(self.start_epoch,self.epochs):
            self.model.train()
            print(('\n' + '%10s' * 11) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'soft', 'rratio', 'targets', 'img_size', 'lr'))

            mloss = torch.zeros(4).to(device)  # mean losses
            msoft_target = torch.zeros(1).to(device)

            pbar = tqdm(enumerate(self.dataloader), total=nb)  # progress bar
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * self.cur_epoch# number integrated batches (since train start)
                x.append(ni)
                y.append(self.optimizer.param_groups[0]['lr']*100)
                self.lr= self.optimizer.param_groups[0]['lr']
                if self.cur_epoch < config.warm_up:
                    self.lr , self.optimizer= self.LR_Scheduler.warmup_schl(self.optimizer, ni, nb)
                imgs = imgs.to(device)
                targets = targets.to(device)
                # Multi-Scale training
                if self.multi_scale:
                    if ni / self.accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                        self.img_size = random.randrange(self.img_sz_min, self.img_sz_max + 1) * 32
                    sf = self.img_size / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / 32.) * 32 for x in
                              imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Plot images with bounding boxes make sure thfe labels are correct
                if ni == 0:
                    fname = 'train_batch%g.jpg' % i
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                    if self.tb_writer:
                        self.tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

                # Run model
                pred = self.model(imgs)

                loss, loss_items = compute_loss(pred, targets, self.model)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results
                # TODO implement distillation
                # if fintune:
                soft_target = 0
                reg_ratio = 0  # 表示有多少target的回归是不如老师的，这时学生会跟gt再学习

                # Compute gradient
                if mixed_precision:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                #update BN weights
                if self.sparse:
                    self.BNsp.update_BN(self.model)
                    bn_weight = self.BNsp.draw_bn(self.model)
                    b_weight.append(bn_weight)
                # Accumulate gradient for x batches before optimizing
                if ni % self.accumulate == 0:
                    self.optimizer.step()  # 更新梯度
                    self.optimizer.zero_grad()

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                msoft_target = (msoft_target * i + soft_target) / (i + 1)
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 9) % (
                    '%g/%g' % (self.cur_epoch, self.epochs - 1), '%.3gG' % mem, *mloss, msoft_target, reg_ratio,
                    len(targets), self.img_size, self.lr)
                pbar.set_description(s)
                # end batch ------------------------------------------------------------------------------------------------

            # Process epoch results
            final_epoch = self.cur_epoch + 1 == self.epochs
            # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
            with torch.no_grad():
                results, maps = test.test(self.cfg,
                                          self.data,
                                          batch_size=self.batch_size,
                                          img_size=opt.img_size,
                                          model=self.model,
                                          conf_thres=0.001 if final_epoch and self.cur_epoch > 0 else 0.1,
                                          # 0.1 for speed
                                          save_json=final_epoch and self.cur_epoch > 0 and 'coco.data' in self.data,)

            # Write epoch results
            self.write_txt_result(results, s)

            # train csv
            self.write_csv(results, *mloss)

            # Write Tensorboard results
            self.write_tensorboard(results, msoft_target, *mloss)

            # Update best mAP
            self.update_map(results)

            # update optimizer
            self.lr_scheduler.step()

            # Save training results
            self.save_model(final_epoch)

            #write bn txt
            if self.sparse:
                self.BNsp.write_txt(self.model, self.tb_writer, self.cur_epoch)

            # draw lr graph
            if self.cur_epoch > opt.epochs - 2 and self.sparse:
                # print(x)
                # print(y)
                plt.figure(figsize=(10, 8), dpi=200)
                plt.xlabel('batch stop')
                plt.ylabel('learning rate')
                plt.plot(x, y, color='r', linewidth=2.0, label='lr')
                plt.plot(x, b_weight,label='sparse')
                plt.legend(loc='upper right')
                plt.savefig('{}/lr_sparse.png'.format(self.train_dir))
                plt.cla()
                # plt.show()
            print(self.best_fitness)
            # end epoch ----------------------------------------------------------------------------------------------------
        if opt.finetune:
            csv_path = os.path.join("prune_result", opt.weights.split("/")[1])
            exist = os.path.exists(os.path.join(csv_path, 'prune.csv'))
            model_name = opt.weights.split('/')[-2] + '_finetune'
            with open(os.path.join(csv_path, 'prune.csv'), 'a+') as f:
                print(os.path.join(csv_path, 'prune.csv'))
                f_csv = csv.writer(f)
                if not exist:
                    title = [
                        'model', 'mAP', 'para', 'time'
                    ]
                    f_csv.writerow(title)
                info_list = [model_name, self.best_fitness, '', '']
                print('fitness',self.best_fitness)
                f_csv.writerow(info_list)
        # end training
        train_time = (time.time() - t0) / 3600
        print('%g epochs completed in %.3f hours.\n' % (self.cur_epoch - self.start_epoch + 1, train_time))
        # draw graph
        draw_graph(self.cur_epoch - self.start_epoch + 1, self.train_loss_ls, self.val_loss_ls, self.prmf_ls,
                   self.train_dir)
        #write pruning bn tensorboard
        if self.sparse:
            self.BNsp.write_tensorboard_after(self.model, self.tb_writer)
        if not opt.finetune:
            self.write_whole_csv(train_time, final_epoch)
            plot_results(self.train_dir)  # save as results.png

        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        return results


if __name__ == '__main__':
    device = torch_utils.select_device(opt.device, apex=mixed_precision)
    # try:
    trainer = Trainer()
    trainer.train()  # train normally
    # except  :
    #     if os.path.exists('error.txt'):
    #         os.remove('error.txt')
    #     with open('error.txt','a+') as f:
    #         f.write(opt.expID)
    #         f.write('\n')
    #         f.write('----------------------------------------------\n')
    #         traceback.print_exc(file=f)
