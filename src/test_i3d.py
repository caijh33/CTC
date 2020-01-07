import argparse
import time

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from net.bilinear_i3d import I3D

from dataset.ucf101_dataset import I3dDataSet

# from net import TSN
import videotransforms
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('--dataset', type=str, default='hmdb51', choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--mode', default='rgb', type=str, choices=['rgb', 'flow'])
parser.add_argument('--test_list', default='data/hmdb51/hmdb51_rgb_val_split_1.txt', type=str)
parser.add_argument('--weights', default='checkpoints/hmdb51/73.202_rgb_model_best.pth.tar', type=str)
parser.add_argument('--arch', type=str, default="inception")
parser.add_argument('--save_scores', type=str, default="test_output/")
parser.add_argument('--test_clips', type=int, default=10)
parser.add_argument('--clip_size', type=int, default=64)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


# ******************************************************************************************
# More need to do: add error analyse model.
# Record which sample is recognition error and rely it.
#
# ******************************************************************************************
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weight_transform(model_dict, pretrain_dict):
    '''

    :return:
    '''
    weight_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(weight_dict)
    return model_dict


def plot_confuse_matrix(matrix, classes,
                        normalize=True,
                        title=None,
                        cmap=plt.cm.Blues
                        ):
    """
    :param matrix:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    return ax


def get_action_index():
    action_label = []
    with open('data/hmdb51/hmdb51_classInd.txt') as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    for line in content:
        label, action = line.split(' ')
        action_label.append(action)
    return action_label


def plot_matrix_test():
    classes = get_action_index()
    confuse_matrix = np.load("test_output/hmdb51_confusion.npy")
    plot_confuse_matrix(confuse_matrix, classes)
    plt.show()


def main():
    if args.dataset == 'ucf101':
        num_class = 101
        data_length = 250
        image_tmpl = "frame{:06d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        data_length = 250
        image_tmpl = "img_{:05d}.jpg"
    elif args.dataset == 'kinetics':
        num_class = 400
        data_length = 250
        image_tmpl = "frame{:06d}.jpg"
    # else:
    # raise ValueError('Unknown dataset '+ args.dataset)

    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_transforms = transforms.Compose([videotransforms.CornerCrop(224)])
    net = I3D(num_classes=num_class, modality=args.mode, dropout_prob=args.dropout)
    checkpoint = torch.load(args.weights)
    # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    val_dataset = I3dDataSet("", args.test_list, num_segments=1,
                             new_length=data_length,
                             modality=args.mode,
                             test_mode=True,
                             dataset=args.dataset,
                             image_tmpl=image_tmpl if args.mode in ["rgb", 'RGB',
                                                                    "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                             random_shift=False,
                             transform=test_transforms)
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.batch_size,
                                              pin_memory=True)
    '''
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))
    '''
    net = torch.nn.DataParallel(net).cuda()
    net.eval()

    data_gen = enumerate(data_loader)
    total_num = len(data_loader.dataset)  # 3783
    max_num = len(data_loader.dataset)

    print("total test num", total_num)

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def plot_point(a, b):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(a, b, 'o')
        plt.title("point figure")
        plt.show()

    def eval_video(video_data):
        '''
        average 10 clips, do it later
        '''
        i, datas, label = video_data
        # data length is 250, get 20 clips and get the average result
        output = None
        # print(len(datas))  # 1 x 3 x 250 x 224 x 224
        for data in datas:
            # print(data.size())
            # data = torch.unsqueeze(data, 0)
            # input_var = torch.autograd.Variable(data, volatile=True)
            # output = net(input_var).data.cpu().numpy().copy()
            for i in range(args.test_clips):
                # print(data.size())
                clip_data = data[:, :, 10 * i:10 * i + args.clip_size, :, :]
                input_var = torch.autograd.Variable(clip_data, volatile=True)
                if output is None:
                    output = net(input_var).data.cpu().numpy().copy() / args.test_clips * 5
                else:
                    output += net(input_var).data.cpu().numpy().copy() / args.test_clips * 5

        '''
        for i in range(args.test_segments):
            input_var = torch.autograd.Variable(data[i], volatile=True)
            output, out_logit = net(input_var) # output: 1 x 101
            softmax = torch.nn.Softmax(dim=1)
            output = softmax(out_logit)
            if i == 0:
                outputs = output / args.test_segments
            else:
                outputs = outputs + output/args.test_segments

        prec1, prec5 = accuracy(outputs.data, target, topk=(1,5))
        '''

        return output, label

    output = []
    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        proc_start_time = time.time()
        rst = eval_video((i, data, label))
        output.append(rst)
        cnt_time = time.time() - proc_start_time
        if i % 10 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                            total_num,
                                                                            float(cnt_time) / (i + 1)))
    # =====output: every video's num and every video's label
    # =====x[0]:softmax value x[1]:label
    video_pred = [np.argmax(x[0]) for x in output]
    np.save("test_output/video_pred.npy", video_pred)
    video_labels = [x[1] for x in output]
    np.save("test_output/video_labels.npy", video_labels)
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    np.save("test_output/" + args.dataset + "_confusion.npy", cf)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    print(cls_acc)
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    if args.save_scores is not None:
        # reorder before saving
        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e: i for i, e in enumerate(sorted(name_list))}
        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)
        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]
        np.savez(args.save_scores + args.dataset + '_' + args.mode + '_' + 'save_scores', scores=reorder_output, labels=reorder_label)


if __name__ == '__main__':
    main()
    plot_matrix_test()
