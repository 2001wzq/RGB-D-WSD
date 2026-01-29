

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=50,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-3,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=4,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=800,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.05,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=25,    help='every n epochs decay learning rate')
parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints')
parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')


parser.add_argument('--rgb_label_root',      type=str, default='./Data/TrainDataset/mydata/RGB/',           help='the training rgb images root')
parser.add_argument('--depth_label_root',    type=str, default='./Data/TrainDataset/mydata/depth/',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str, default='./Data/TrainDataset/mydata/GT/',            help='the training gt images root')
parser.add_argument('--val_rgb_root',        type=str, default='./Data/TestDataset/mydata/RGB/',      help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str, default='./Data/TestDataset/mydata/depth/',    help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str, default='./Data/TestDataset/mydata/GT/',       help='the test gt images root')
parser.add_argument('--save_path',           type=str, default='./Checkpoint/GATConv_layers_2_800/',    help='the path to save models and logs')


opt = parser.parse_args()



