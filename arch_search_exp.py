import os
import os.path as osp
import sys
import time
import numpy as np
import math
sys.path.append('./')

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import get_args
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.datasets.concatdataset import ConcatDataset
from lib.loss import SequenceCrossEntropyLoss
from lib.utils.logging import Logger
from lib.utils.meters import AverageMeter
from lib.models.proxyless import ProxylessBackbone
from lib.utils.serialization import save_checkpoint
from lib.models.mix_ops import MixedEdge


def get_data(data_dir, voc_type, max_len, num_samples, 
             height, width, batch_size, workers, is_train, keep_ratio):
    if isinstance(data_dir, list):
        dataset_list = []
        for data_dir_ in data_dir:
            dataset_list.append(LmdbDataset(data_dir_, voc_type, max_len, num_samples))
            dataset = ConcatDataset(dataset_list)
    else:
        dataset = LmdbDataset(data_dir, voc_type, max_len, num_samples)
    print('total image: ', len(dataset))

    if is_train:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=False,
        collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True, drop_last=False,
        collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
    return dataset, data_loader


class ArchSearchManager:

    def __init__(self, net, train_dataset, train_loader, eval_dataset, eval_loader, 
                 global_args):
        self.net = net
        self.global_args = global_args

        self.cnn_encoder.init_arch_params(
            init_type='uniform',
            init_ratio=1e-3
        )

        self.weight_optimizer = self.build_weight_optimizer(
            self.get_weight_parameters())
        self.arch_optimizer = self.build_arch_optimizer(
            self.get_architecture_paramters())

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.loss_weights = {'loss_rec': 1.0}
        self.update_arch_param_every = global_args.update_arch_param_every
        self.grid_update_step = global_args.grid_update_step
        self.print_freq = 50
        self.save_freq = 500

        self._eval_iter = None
        self.warmup_epoch = global_args.warmup_epoch
        self.binary_mode = global_args.binary_mode

    @property
    def cnn_encoder(self) -> ProxylessBackbone:
        return self.net.module.encoder

    def get_architecture_paramters(self):
        for name, params in self.net.named_parameters():
            if 'AP_path_alpha' in name:
                yield params

    def get_weight_parameters(self):
        for name, params in self.net.named_parameters():
            if 'AP_path_alpha' not in name:
                yield params


    def build_arch_optimizer(self, arch_params):
        arch_optimizer = torch.optim.Adam(
            arch_params, lr=1e-3, betas=(0, 0.999), eps=1e-8, weight_decay=0)
        return arch_optimizer

    def build_weight_optimizer(self, weight_params):
        weight_optimizer = torch.optim.Adadelta(
            weight_params, lr=0.9, weight_decay=5e-4
        )
        return weight_optimizer

    def _parse_data(self, inputs):
        imgs, label_encs, lengths = inputs
        input_dict = {
            'images': imgs.cuda(),
            'rec_targets': label_encs.cuda(),
            'rec_lengths': lengths
        }
        return input_dict

    def _forward(self, input_dict):
        output_dict = self.net(input_dict)
        return output_dict

    def get_update_scheduler(self, n_batch):
        schedule = {}
        for i in range(n_batch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.grid_update_step
        return schedule

    def start_search(self, start_epoch=0):
        n_arch_params  = len(list(self.cnn_encoder.architecture_parameters()))
        n_binary_gates = len(list(self.cnn_encoder.binary_gates()))
        n_weight_param = len(list(self.cnn_encoder.weight_parameters()))
        print('#arch_params: %d\t#binary_gates: %d\t#n_weight_params: %d' % (
            n_arch_params, n_binary_gates, n_weight_param
        ))
        self.net.train()
        n_batch = len(self.train_loader)
        update_scheduler = self.get_update_scheduler(n_batch)
        for epoch in range(start_epoch, self.global_args.epochs):
            entropys = AverageMeter()
            losses = AverageMeter()
            arch_losses = AverageMeter()
            reg_losses = AverageMeter()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            end = time.time()
            for i, inputs in enumerate(self.train_loader):
                data_time.update(time.time() - end)
                net_entropy = self.cnn_encoder.entropy()
                entropys.update(net_entropy.item() / n_arch_params, 1)

                # Update weight
                self.cnn_encoder.reset_binary_gates()
                self.cnn_encoder.unused_modules_off()
                input_dict  = self._parse_data(inputs)
                output_dict = self._forward(input_dict)

                batch_size = input_dict['images'].size(0)
                total_loss = 0
                for k, loss in output_dict['losses'].items():
                    loss = loss.mean(dim=0, keepdim=True)
                    total_loss += self.loss_weights[k] * loss

                losses.update(total_loss.item(), batch_size)
                self.net.zero_grad()
                total_loss.backward()
                self.weight_optimizer.step()
                self.cnn_encoder.unused_modules_back()

                batch_time.update(time.time() - end)

                if epoch >= self.warmup_epoch:
                    warmup_up = False
                    for j in range(update_scheduler.get(i, 0)):
                        arch_loss, reg_loss = self.gradient_step()
                        arch_losses.update(arch_loss.item(), batch_size)
                        reg_losses.update(reg_loss.item(), batch_size)
                else:
                    warmup_up = True

                if i % self.print_freq == 0:
                    print('%s [%d][%d/%d]\t'
                        'TLoss %.3f (%.3f)\t'
                        'VLoss %.3f (%.3f)\t'
                        'RegLoss: %.4f (%.4f)\t'
                        'Time %.3f (%.3f)\t'
                        'Data %.3f (%.3f)\t'
                        'Entr %.5f (%.5f)' % ( 'Train' if not warmup_up else 'Warmup',
                            epoch, i, n_batch - 1, 
                            losses.val, losses.avg, 
                            arch_losses.val, arch_losses.avg,
                            reg_losses.val, reg_losses.avg,
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            entropys.val, entropys.avg))

                if i % self.save_freq == 0:
                    save_checkpoint(
                        {
                            'warmup': warmup_up,
                            'epoch': epoch,
                            'state_dict': self.net.state_dict(),
                            'weight_optimizer': self.weight_optimizer.state_dict(),
                            'arch_optimizer': self.arch_optimizer.state_dict()
                        }, is_best=False, 
                        fpath=os.path.join(args.logs_dir, 'checkpoint.pth.tar'))
                    for idx, block in enumerate(self.cnn_encoder.blocks):
                        print('%d. %s' % (idx, block.module_str), end='\t')
                        prob_list = ['%.3f' % x for x in block.mobile_inverted_conv.probs_over_ops.data.cpu().numpy().tolist()]
                        print('# %s' % (str(prob_list)))
                end = time.time()

    def next_eval_batch(self):
        if self._eval_iter is None:
            self._eval_iter = iter(self.eval_loader)
        try:
            data = next(self._eval_iter)
        except StopIteration:
            self._eval_iter = iter(self.eval_loader)
            data = next(self._eval_iter)
        return data

    def gradient_step(self):
        MixedEdge.MODE = self.binary_mode
        eval_inputs = self.next_eval_batch()
        eval_inputs_dict = self._parse_data(eval_inputs)
        output_dict = self._forward(eval_inputs_dict)
        self.cnn_encoder.reset_binary_gates()
        self.cnn_encoder.unused_modules_off()

        batch_size = eval_inputs_dict['images'].size(0)
        ce_loss = 0
        for k, loss in output_dict['losses'].items():
            loss = loss.mean(dim=0, keepdim=True)
            ce_loss += self.loss_weights[k] * loss

        if self.global_args.add_flops_regularization_loss:
            reg_alpha = self.global_args.flops_reg_alpha
            reg_belta = self.global_args.flops_reg_belta
            flops_ref_value = self.global_args.flops_ref_value
            
            input_x = torch.zeros([1, 3, 32, 100]).cuda()
            e_flops = self.cnn_encoder.expected_flops(input_x)
            reg_loss = (torch.log(e_flops) / math.log(flops_ref_value)) ** reg_belta
            total_loss = reg_alpha * ce_loss * reg_loss
        else:
            reg_loss = torch.zeros([1])
            total_loss = ce_loss

        self.net.zero_grad()
        total_loss.backward()
        self.cnn_encoder.set_arch_param_grad()
        self.arch_optimizer.step()
        if MixedEdge.MODE == 'two':
            self.cnn_encoder.rescale_updated_arch_param()
        self.cnn_encoder.unused_modules_back()
        MixedEdge.MODE = None
        return ce_loss, reg_loss


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    cfg_save_path = osp.join(args.logs_dir, 'cfg.txt')
    cfgs = vars(args)
    with open(cfg_save_path, 'w') as f:
        for k, v in cfgs.items():
            f.write('{}: {}\n'.format(k, v))
    
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)
    train_dataset, train_loader = get_data(
        args.train_data_dir, args.voc_type, args.max_len, args.num_train,
        args.height, args.width, args.batch_size, args.workers, True, args.keep_ratio)
    eval_dataset, eval_loader = get_data(
        args.eval_data_dir, args.voc_type, args.max_len, args.num_eval,
        args.height, args.width, args.batch_size, args.workers, True, args.keep_ratio)
    assert train_dataset is not None and eval_dataset is not None 
    rec_num_classes = train_dataset.rec_num_classes
    max_len = train_dataset.max_len
    eos = train_dataset.char2id[train_dataset.EOS]

    print('arch: ', args.arch)
    model = ModelBuilder(arch=args.arch, rec_num_classes=rec_num_classes,
                        sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=max_len,
                        eos=eos,  args=args, STN_ON=args.STN_ON)
    model = model.cuda()
    model = nn.DataParallel(model)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    manager = ArchSearchManager(
        model, train_dataset, train_loader, eval_dataset, eval_loader, args)
    manager.start_search()
