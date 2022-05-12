import argparse

import configargparse

def str2bool(v):
    """Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def get_args():
    parser = configargparse.ArgParser(description='PSMNet')
    parser.add('-c', '--config',
               default='/home/mcokelek21/Desktop/Github/PseudoLiDAR523/Pseudo_Lidar_V2/src/configs/sdn_kitti_train.config',
               is_config_file=True, help='config file')

    parser.add_argument('--save_path', type=str, default='',
                        help='path to save the log, tensorbaord and checkpoint')
    # network
    parser.add_argument('--data_type', default='depth', choices=['disparity', 'depth'],
                        help='the network can predict either disparity or depth')
    parser.add_argument('--arch', default='SDNet', choices=['SDNet', 'PSMNet'],
                        help='Model Name, default: SDNet.')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity, the range of the disparity cost volume: [0, maxdisp-1]')
    parser.add_argument('--down', type=float, default=2,
                        help='reduce x times resolution when build the depth cost volume')
    parser.add_argument('--maxdepth', type=int, default=80,
                        help='the range of the depth cost volume: [1, maxdepth]')
    # dataset
    parser.add_argument('--kitti2015', action='store_true',
                        help='If false, use 3d kitti dataset. If true, use kitti stereo 2015, default: False')
    parser.add_argument('--dataset', default='kitti', choices=['sceneflow', 'kitti'],
                        help='train with sceneflow or kitti')
    parser.add_argument('--datapath', default=None,
                        help='root folder of the dataset')
    parser.add_argument('--split_train', default='split/train.txt',
                        help='data splitting file for training')
    parser.add_argument('--split_val', default='split/subval.txt',
                        help='data splitting file for validation')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--btrain', type=int, default=12,
                        help='training batch size')
    parser.add_argument('--bval', type=int, default=4,
                        help='validation batch size')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of dataset workers')
    # learning rate
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_stepsize', nargs='+', type=int, default=[200],
                        help='drop lr in each step')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='gamma of the learning rate scheduler')
    # resume
    parser.add_argument('--resume',
                        help='path to a checkpoint')
    parser.add_argument('--pretrain', default=None,
                        help='path to pretrained model')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch')
    # evaluate
    parser.add_argument('--evaluate', action='store_true',
                        help='do evaluation')
    parser.add_argument('--calib_value', type=float, default=1017,
                        help='manually define focal length. (sceneflow does not have configuration)')
    parser.add_argument('--dynamic_bs', action='store_true',
                        help='If true, dynamically calculate baseline from calibration file. If false, use 0.54')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='evaluate model every n epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='save checkpoint every n epoch.')
    parser.add_argument('--generate_depth_map', default=True, action='store_true',
                        help='if true, generate depth maps and save the in save_path/depth_maps/{data_tag}/')
    parser.add_argument('--data_list',
                        default='',
                        help='generate depth maps for all the data in this list')
    parser.add_argument('--data_tag', default='val',
                        help='the suffix of the depth maps folder')

    # debug
    parser.add_argument('--debug', type=str2bool, default=False, help='created by Mert. For debugging.')
    args = parser.parse_args()
    return args
