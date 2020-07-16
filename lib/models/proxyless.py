

from queue import Queue
import re
import numpy as np

import torch
import torch.nn as nn

from .mix_ops import MixedEdge, build_candidate_ops, conv_func_by_name
from .layers import MBInvertedConvLayer, IdentityLayer, ZeroLayer, MobileInvertedResidualBlock, count_conv_flop

import sys
from config import get_args
global_args = get_args(sys.argv[1:])


class NasRecBackbone(nn.Module):

    def __init__(self, first_conv, blocks, with_lstm):
        super(NasRecBackbone, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)

        # with_lstm = False
        self.with_lstm  = with_lstm
        if with_lstm:
            self.rnn = nn.LSTM(
                512, 256, bidirectional=True, num_layers=2, batch_first=True)
            self.out_planes = 256 * 2
        else:
            self.out_planes = 512

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        cnn_feat = x.squeeze(dim=2)
        cnn_feat = cnn_feat.transpose(2, 1)
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat
    
    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    @property
    def config(self):
        return {
            'name': NasRecBackbone.__name__,
            'bn': self.get_bn_param(),
            'first_conv': 'conv_in3_out32_k3_s1_p1',
            'blocks': [
                block.config for block in self.blocks
            ]
        }

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    
    @staticmethod
    def build_from_config(config):
        first_conv_config = config['first_conv']
        match_obj = re.match(r'conv_in(\d+)_out(\d+)_k(\d+)_s(\d+)_p(\d+)',
                             first_conv_config)
        in_channel = int(match_obj.group(1))
        out_channel = int(match_obj.group(2))
        kernel_size = int(match_obj.group(3))
        stride = int(match_obj.group(4))
        padding = int(match_obj.group(5))
        first_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for block_config in config['blocks']:
            blocks.append(
                MobileInvertedResidualBlock.build_from_config(block_config)
            )
        net = NasRecBackbone(first_conv, blocks)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        return net


class CompactRecBackbone(NasRecBackbone):

    def __init__(self, with_lstm=True, **kwargs):
        first_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        if global_args.stride_stages == '':
            stride_stages = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]
        else:
            stride_stages = eval(global_args.stride_stages)
        if global_args.n_cell_stages == '':
            n_cell_stages = [3, 3, 3, 3, 3]
        else:
            n_cell_stages = eval(global_args.n_cell_stages)

        width_stages = [32, 64, 128, 256, 512]
        conv_candidates = [
            '5x5_MBConv1', '5x5_MBConv3', '5x5_MBConv6', 
            '3x3_MBConv1', '3x3_MBConv3', '3x3_MBConv6',
            'Zero'
        ]

        print('### CompactRecBackbone: ')
        print('%s' % str(stride_stages))
        if global_args.conv_op_ids != "":
            conv_op_ids = np.array(eval(global_args.conv_op_ids))
        else:
            conv_op_ids = [5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 4, 3, 4, 6, 6]
        for op_id in conv_op_ids:
            print(conv_candidates[op_id])
        print('###\n')
        # import ipdb; ipdb.set_trace()
        assert len(conv_op_ids) == sum(n_cell_stages)
        blocks = []
        input_channel = 32
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = (1, 1)
                block_i = len(blocks)
                conv_op = conv_func_by_name(conv_candidates[conv_op_ids[block_i]])(input_channel, width, stride)
                if stride == (1, 1) and input_channel == width:
                    shortcut = IdentityLayer()
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width
        self.out_channel = input_channel

        super(CompactRecBackbone, self).__init__(first_conv, blocks, with_lstm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

class ProxylessBackbone(NasRecBackbone):

    def __init__(self, with_lstm=True, bn_param=(0.1, 1e-3), **kwargs):
        self._redundant_modules = None
        self._unused_modules = None
        in_channel = 3
        first_conv = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        conv_candidates = [
            '5x5_MBConv1', '5x5_MBConv3', '5x5_MBConv6',
            '3x3_MBConv1', '3x3_MBConv3', '3x3_MBConv6',
        ]
        if global_args.stride_stages == '':
            stride_stages = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]
        else:
            stride_stages = eval(global_args.stride_stages)
        if global_args.n_cell_stages == '':
            n_cell_stages = [3, 3, 3, 3, 3]
        else:
            n_cell_stages = eval(global_args.n_cell_stages)
        width_stages = [32, 64, 128, 256, 512]

        print('### ProxylessBackbone ###')
        print('stride_stages: ', stride_stages)
        print('n_cell_stages: ', n_cell_stages)
        print(' width_stafes: ', width_stages)
        print('### ### ### ###')

        blocks = []
        input_channel = 32
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = (1, 1)
                if stride == (1, 1) and input_channel == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride, 'weight_bn_act',
                ))
                if stride == (1, 1) and input_channel == width:
                    shortcut = IdentityLayer()
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width
        self.out_channel = input_channel

        super(ProxylessBackbone, self).__init__(first_conv, blocks, with_lstm)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = bn_param[0]
                m.eps = bn_param[1]

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param
    
    def binary_gates(self):
        for name, params in self.named_parameters():
            if 'AP_path_wb' in name:
                yield params

    def weight_parameters(self):
        for name, params in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield params

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith("MixedEdge"):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')
    
    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support set_arch_param_grad()')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support rescale_updated_arch_param()')
    
    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support set_chosen_op_active')

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return NasRecBackbone(self.first_conv, list(self.blocks))

    def expected_flops(self, x):
        expected_flops = 0
        # first conv
        flop = count_conv_flop(self.first_conv[0], x)
        x = self.first_conv(x)
        expected_flops += flop
        # blocks
        for block in self.blocks:
            mb_conv = block.mobile_inverted_conv
            assert isinstance(mb_conv, MixedEdge)
            if block.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = block.shortcut.get_flops(x)
            expected_flops += shortcut_flop
            probs_over_ops = mb_conv.current_prob_over_ops
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                op_flops, _ = op.get_flops(x)
                expected_flops += op_flops * probs_over_ops[i]
            x = block(x)
        return expected_flops


if __name__ == "__main__":
    net = ProxylessBackbone()
    import ipdb; ipdb.set_trace()
