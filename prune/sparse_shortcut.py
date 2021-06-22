import argparse

from models import *
from utils.prune_utils import *
from utils.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--percent_max', type=int, default=95, help='channel prune percent')
    parser.add_argument('--percent_min', type=int, default=40, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    per_min, per_max = opt.percent_min, opt.percent_max
    sparse_file_path = os.path.join("/".join(opt.weights.split("/")[:-2]), "sparse_shortcut_result.csv")
    if_exist = os.path.exists(sparse_file_path)
    sparse_file = open(sparse_file_path, "a+")
    if not if_exist:
        title = "Model_name,"+",".join(map(lambda x: str(x), range(per_min, per_max+1)))
        sparse_file.write(title)
        sparse_file.write("\n")

    if opt.weights.endswith(".pt"):
        model_name = opt.weights.split("/")[-1][:-3]
    elif opt.weights.endswith(".pth"):
        model_name = opt.weights.split("/")[-1][:-4]
    elif opt.weights.endswith(".weights"):
        model_name = opt.weights.split("/")[-1][:-8]
    else:
        raise ValueError("Wrong model name")

    percent_ls = [num/100 for num in range(opt.percent_min, opt.percent_max+1, 1)]
    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ', opt.weights)



    CBL_idx, Conv_idx, prune_idx,shortcut_idx,shortcut_all= parse_module_defs2(model.module_defs)

    sort_prune_idx = [idx for idx in prune_idx if idx not in shortcut_idx]

    #将所有要剪枝的BN层的α参数，拷贝到bn_weights列表
    bn_weights = gather_bn_weights(model.module_list, sort_prune_idx)

    #torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
    sorted_bn = torch.sort(bn_weights)[0]


    #避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for idx in sort_prune_idx:
        #.item()可以得到张量里的元素值
        highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
    highest_thre = min(highest_thre)

    # 找到highest_thre对应的下标对应的百分比
    # percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)

    print(f'Suggested Threshold should be less than {highest_thre:.4f}.')
    # print(f'The corresponding prune ratio is {percent_limit:.3f},but you can set higher.')


    def prune_and_eval(model, sorted_bn, percent=.0):
        model_copy = deepcopy(model)
        thre_index = int(len(sorted_bn) * percent)
        #获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
        thre1 = sorted_bn[thre_index]

        print(f'When percent is {percent:.4f}, channels with Gamma value less than {thre1:.6f} are pruned!')

        remain_num = 0
        idx_new=dict()
        for idx in prune_idx:
            
            if idx not in shortcut_idx:
                
                bn_module = model_copy.module_list[idx][1]

                mask = obtain_bn_mask(bn_module, thre1)
                #记录剪枝后，每一层卷积层对应的mask
                # idx_new[idx]=mask.cpu().numpy()
                idx_new[idx]=mask
                remain_num += int(mask.sum())
                bn_module.weight.data.mul_(mask)
                #bn_module.bias.data.mul_(mask*0.0001)
            else:
                bn_module = model_copy.module_list[idx][1]

                mask=idx_new[shortcut_idx[idx]]
                idx_new[idx]=mask
     
                remain_num += int(mask.sum())
                bn_module.weight.data.mul_(mask)

        print(f'When percent is {percent:.4f}, number of channels has been reduced from {len(sorted_bn)} to '
              f'{remain_num}, and the prune ratio: {1-remain_num/len(sorted_bn):.3f}')

        return thre1

    model_res = opt.weights.split("/")[-2] + "-" + model_name + ","
    for percent in percent_ls:
        threshold = prune_and_eval(model, sorted_bn, percent)
        print("{}---->{}".format(percent, threshold))
        model_res += str(threshold.tolist())
        model_res += ","
    sparse_file.write(model_res+'\n')

