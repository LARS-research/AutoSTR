from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import sys
try:
    from config import get_args
    global_args = get_args(sys.argv[1:])
except:
    print('run local')


class Identity(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class MBInvertedConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(1, 1), expand_ratio=6, mid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        feature_dims = int(self.in_channels * self.expand_ratio) if mid_channels is None else mid_channels
        if self.expand_ratio == 1:
            self.inverted_bottleneck = nn.Sequential()
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dims, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dims)),
                ('act', nn.ReLU6(inplace=True)),
            ]))
        pad = self.kernel_size // 2
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dims, feature_dims, kernel_size, stride, pad, groups=feature_dims, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dims)),
            ('act', nn.ReLU6(inplace=True))
        ]))
        self.point_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dims, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MixedMobileConvs(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        ops = [
            MBInvertedConvLayer(in_channels, out_channels, 3, stride, 1),
            MBInvertedConvLayer(in_channels, out_channels, 3, stride, 3),
            MBInvertedConvLayer(in_channels, out_channels, 3, stride, 6),
            MBInvertedConvLayer(in_channels, out_channels, 5, stride, 1),
            MBInvertedConvLayer(in_channels, out_channels, 5, stride, 3),
            MBInvertedConvLayer(in_channels, out_channels, 5, stride, 6),
            Identity(),
        ]
        self._downsample = (stride != [1, 1])
        if self._downsample or global_args.remove_skip:
            self._n_choices = len(ops) - 1
        else:
            self._n_choices = len(ops)
        self._ops = nn.ModuleList(ops)
        self._active_op = 0
        self._stride = stride

    @property
    def active_op(self):
        return self._active_op

    @active_op.setter
    def active_op(self, op_id):
        assert op_id < self._n_choices
        self._active_op = op_id

    def forward(self, x):
        choice_op = self._ops[self.active_op]
        if isinstance(choice_op, Identity):
            return x
        else:
            shortcut = x
            x = choice_op(x)
            if not self._downsample:
                return shortcut + x
            return x


class MixedMobileConvs_Compact(nn.Module):

    def __init__(self, in_channels, out_channels, stride, op_choice):
        super().__init__()
        if op_choice == 0:
            conv = MBInvertedConvLayer(in_channels, out_channels, 3, stride, 1)
        elif op_choice == 1:
            conv = MBInvertedConvLayer(in_channels, out_channels, 3, stride, 3)
        elif op_choice == 2:
            conv = MBInvertedConvLayer(in_channels, out_channels, 3, stride, 6)
        elif op_choice == 3:
            conv = MBInvertedConvLayer(in_channels, out_channels, 5, stride, 1)
        elif op_choice == 4:
            conv = MBInvertedConvLayer(in_channels, out_channels, 5, stride, 3)
        elif op_choice == 5:
            conv = MBInvertedConvLayer(in_channels, out_channels, 5, stride, 6)
        elif op_choice == 6:
            conv = Identity()
        else:
            raise NotImplementedError
        self.conv = conv
        self._downsample = (stride != [1, 1])
    
    def forward(self, x):
        if isinstance(self.conv, Identity):
            return x
        else:
            shortcut = x
            x = self.conv(x)
            if not self._downsample:
                return shortcut + x
            return x


class SuperNet_MBConvs_Compact(nn.Module):

    def __init__(self, with_lstm=False, **kwargs):
        super().__init__()
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # from str config to list
        try:
            self.path = np.array(eval(global_args.path))
            self.op_choices = eval(global_args.op_choices)
        except:
            self.path = np.array(eval("[[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 1], [2, 1], [2, 1], [3, 1], [3, 1], [4, 1], [4, 1], [4, 1], [4, 1], [5, 2], [5, 2]]"))
            self.op_choices = eval('[4, 5, 5, 2, 2, 5, 4, 5, 1, 5, 2, 1, 3, 4, 5]')
        print('\nEval config: ')
        print(self.path)
        print(self.op_choices)
        print('*' * 40)
        import ipdb; ipdb.set_trace()
        features = []
        for i in range(1, len(self.path)):
            pre_node = self.path[i - 1]
            cur_node = self.path[i]
            in_channels = self.get_channel(pre_node)
            out_channels = self.get_channel(cur_node)
            gap = cur_node - pre_node
            if (gap == np.array([0, 0])).all():
                stride = [1, 1]
            elif (gap == np.array([1, 0])).all():
                stride = [2, 1]
            elif (gap == np.array([1, 1])).all():
                stride = [2, 2]
            else:
                raise NotImplementedError
            op_choice = self.op_choices[i - 1]
            features.append(
                MixedMobileConvs_Compact(in_channels, out_channels, stride, op_choice))
        self.features = nn.Sequential(*features)
        self.with_lstm = with_lstm
        if with_lstm:
            self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
            self.out_planes = 256 * 2
        else:
            self.out_planes = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, choice=None):
        x = self.stem_conv(x)
        x = self.features(x)
        cnn_feat = x.squeeze(dim=2)
        cnn_feat = cnn_feat.transpose(2, 1)
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat
    
    def get_channel(self, node):
        x = node[0]
        if x == 0:
            return 32
        else:
            return 32 * (2 ** (x - 1))


class SuperNet_MBConvs(nn.Module):

    def __init__(self, with_lstm=False, **kwargs):
        super().__init__()
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # self.path = np.array([
        #     [0, 0], [0, 0], [0, 0], [0, 0],
        #     [1, 0],
        #     [2, 1], [2, 1], [2, 1],
        #     [3, 1], [3, 1],
        #     [4, 1], [4, 1], [4, 1], [4, 1],
        #     [5, 2], [5, 2]
        # ])
        self.path = np.array(eval(global_args.path))
        print('Build SuperNet_MBConvs with path: ', global_args.path)
        valid_op_choice = []
        edge_str_list = []
        edge_order_dict = OrderedDict()
        strides = []
        for i in range(1, len(self.path)):
            pre_node = self.path[i - 1]
            cur_node = self.path[i]
            in_channels = self.get_channel(pre_node)
            out_channels = self.get_channel(cur_node)
            gap = cur_node - pre_node
            if (gap == np.array([0, 0])).all():
                stride = [1, 1]
            elif (gap == np.array([1, 0])).all():
                stride = [2, 1]
            elif (gap == np.array([1, 1])).all():
                stride = [2, 2]
            else:
                raise ValueError('Valid Path')
            cur_layer = MixedMobileConvs(in_channels, out_channels, stride)
            pre_node_str = '%d-%d-%d' % (pre_node[0], pre_node[1], i - 1)
            cur_node_str = '%d-%d-%d' % (cur_node[0], cur_node[1], i)
            edge_key = '%s$%s' % (pre_node_str, cur_node_str)
            edge_str_list.append(edge_key)
            assert edge_key not in edge_order_dict
            edge_order_dict[edge_key] = cur_layer
            valid_op_choice.append(cur_layer._n_choices)
            strides.append(stride)
        print('Valid op_choices', valid_op_choice)
        self.all_edges = nn.Sequential(edge_order_dict)
        self.all_edges_str_list = edge_str_list
        self.valid_op_choice = valid_op_choice
    
        self.with_lstm = with_lstm
        if with_lstm:
            self.rnn = nn.LSTM(
                512, 256, bidirectional=True, num_layers=2, batch_first=True)
            self.out_planes = 256 * 2
        else:
            self.out_planes = 512
        
    def get_channel(self, node):
        x = node[0]
        if x == 0:
            return 32
        else:
            return 32 * (2 ** (x - 1))

    def forward(self, x, arch=None):
        x = self.stem_conv(x)
        if arch is None:
            arch = [np.random.randint(i) for i in self.valid_op_choice]
        for edge_key, arch_id in zip(self.all_edges_str_list, arch):
            select_edge = self.all_edges._modules[edge_key]
            select_edge.active_op = arch_id
            x = select_edge(x)
        cnn_feat = x.squeeze(dim=2)
        cnn_feat = cnn_feat.transpose(2, 1)
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat


if __name__ == "__main__":
    # net = SuperNet_MBConvs()
    # x = torch.randn(1, 3, 32, 100)
    # print(x.shape)
    # y = net(x)
    # print(y.shape)
    net = SuperNet_MBConvs_Compact()
    x = torch.randn(1, 3, 32, 100)
    from thop import profile
    flops, params = profile(net, inputs=(x, ))
    print('Flops: %.2f G, params: %.2f M' % (flops / 1e9, params/1e6))
    # y = net(x)
    # print(net)
    # print(y.shape)
    

