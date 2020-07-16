from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os.path as osp
import numpy as np
import math
import time
import pickle

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.datasets.concatdataset import ConcatDataset
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from tools.flops_counter import get_model_complexity_info

global_args = get_args(sys.argv[1:])


def get_data(data_dir, voc_type, max_len, num_samples, height, width, batch_size, workers, is_train, keep_ratio, n_max_samples=-1):
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(LmdbDataset(data_dir_, voc_type, max_len, num_samples))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = LmdbDataset(data_dir, voc_type, max_len, num_samples)
  print('total image: ', len(dataset))

  if n_max_samples > 0:
    n_all_samples = len(dataset)
    assert n_max_samples < n_all_samples
    # make sample indices static for every run
    sample_indices_cache_file = '.sample_indices.cache.pkl'
    if os.path.exists(sample_indices_cache_file):
      with open(sample_indices_cache_file, 'rb') as fin:
        sample_indices = pickle.load(fin)
      print('load sample indices from sample_indices_cache_file: ', n_max_samples)
    else:
      sample_indices = np.random.choice(n_all_samples, n_max_samples, replace=False)
      with open(sample_indices_cache_file, 'wb') as fout:
        pickle.dump(sample_indices, fout)
      print('random sample: ', n_max_samples)
    sub_sampler = SubsetRandomSampler(sample_indices)
  else:
    sub_sampler = None

  if is_train:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, sampler=sub_sampler,
      shuffle=(True if sub_sampler is None else False), pin_memory=True, drop_last=True,
      collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  else:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=False, pin_memory=True, drop_last=False,
      collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))

  return dataset, data_loader


def get_dataset(data_dir, voc_type, max_len, num_samples):
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(LmdbDataset(data_dir_, voc_type, max_len, num_samples))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = LmdbDataset(data_dir, voc_type, max_len, num_samples)
  print('total image: ', len(dataset))
  return dataset


def get_dataloader(synthetic_dataset, real_dataset, height, width, batch_size, workers,
                   is_train, keep_ratio):
  num_synthetic_dataset = len(synthetic_dataset)
  num_real_dataset = len(real_dataset)

  synthetic_indices = list(np.random.permutation(num_synthetic_dataset))
  synthetic_indices = synthetic_indices[num_real_dataset:]
  real_indices = list(np.random.permutation(num_real_dataset) + num_synthetic_dataset)
  concated_indices = synthetic_indices + real_indices
  assert len(concated_indices) == num_synthetic_dataset

  sampler = SubsetRandomSampler(concated_indices)
  concated_dataset = ConcatDataset([synthetic_dataset, real_dataset])
  print('total image: ', len(concated_dataset))

  data_loader = DataLoader(concated_dataset, batch_size=batch_size, num_workers=workers,
    shuffle=False, pin_memory=True, drop_last=True, sampler=sampler,
    collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  return concated_dataset, data_loader

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  args.cuda = args.cuda and torch.cuda.is_available()
  if args.cuda:
    print('using cuda.')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type('torch.FloatTensor')
  
  # Redirect print to both console and log file
  if not args.evaluate:
    # make symlink
    if not os.path.exists(args.logs_dir):
      os.makedirs(args.logs_dir)
    make_symlink_if_not_exists(osp.join(args.real_logs_dir, args.logs_dir), osp.dirname(osp.normpath(args.logs_dir)))
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    train_tfLogger = TFLogger(osp.join(args.logs_dir, 'train'))
    eval_tfLogger = TFLogger(osp.join(args.logs_dir, 'eval'))

  # Save the args to disk
  if not args.evaluate:
    cfg_save_path = osp.join(args.logs_dir, 'cfg.txt')
    cfgs = vars(args)
    with open(cfg_save_path, 'w') as f:
      for k, v in cfgs.items():
        f.write('{}: {}\n'.format(k, v))

  # Create data loaders
  if args.height is None or args.width is None:
    args.height, args.width = (32, 100)
  print('height:', args.height, ' width: ', args.width)

  if not args.evaluate: 
    train_dataset, train_loader = \
      get_data(args.train_data_dir, args.voc_type, args.max_len, args.num_train,
               args.height, args.width, args.batch_size, args.workers, True, args.keep_ratio, n_max_samples=args.n_max_samples)
  test_dataset, test_loader = \
    get_data(args.test_data_dir, args.voc_type, args.max_len, args.num_test,
             args.height, args.width, args.batch_size, args.workers, False, args.keep_ratio)

  if args.evaluate:
    max_len = test_dataset.max_len
  else:
    max_len = max(train_dataset.max_len, test_dataset.max_len)
    train_dataset.max_len = test_dataset.max_len = max_len
  # Create model
  model = ModelBuilder(arch=args.arch, rec_num_classes=test_dataset.rec_num_classes,
                       sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=max_len,
                       eos=test_dataset.char2id[test_dataset.EOS], args=args, STN_ON=args.STN_ON)
  #print('model: ', model)
  # import ipdb; ipdb.set_trace()
  params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
  encoder_flops, _ = get_model_complexity_info(model.encoder, input_res=(3, 32, 100), as_strings=False)
  print('num of parameters: ', params_num)
  print('encoder flops: ', encoder_flops)

  # Load from checkpoint
  if args.evaluation_metric == 'accuracy':
    best_res = 0
  elif args.evaluation_metric == 'editdistance':
    best_res = math.inf
  else:
    raise ValueError("Unsupported evaluation metric:", args.evaluation_metric)
  start_epoch = 0
  start_iters = 0
  if args.resume:
    checkpoint = load_checkpoint(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    # compatibility with the epoch-wise evaluation version
    if 'epoch' in checkpoint.keys():
      start_epoch = checkpoint['epoch']
    else:
      start_iters = checkpoint['iters']
      start_epoch = int(start_iters // len(train_loader)) if not args.evaluate else 0
    best_res = checkpoint['best_res']
    print("=> Start iters {}  best res {:.1%}"
          .format(start_iters, best_res))
  
  if args.cuda:
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model)

  # Evaluator
  evaluator = Evaluator(model, args.evaluation_metric, args.cuda)

  if args.evaluate:
    print('Test on {0}:'.format(args.test_data_dir))
    if len(args.vis_dir) > 0:
      vis_dir = osp.join(args.logs_dir, args.vis_dir)
      if not osp.exists(vis_dir):
        os.makedirs(vis_dir)
    else:
      vis_dir = None

    start = time.time()
    evaluator.evaluate(test_loader, dataset=test_dataset, vis_dir=vis_dir)
    print('it took {0} s.'.format(time.time() - start))
    return

  # Optimizer
  param_groups = model.parameters()
  param_groups = filter(lambda p: p.requires_grad, param_groups)
  optimizer = optim.Adadelta(param_groups, lr=args.lr, weight_decay=args.weight_decay)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=eval(args.milestones), gamma=0.1)

  # Trainer
  loss_weights = {}
  loss_weights['loss_rec'] = 1.
  if args.debug:
    args.print_freq = 1
  trainer = Trainer(model, args.evaluation_metric, args.logs_dir, 
                    iters=start_iters, best_res=best_res, grad_clip=args.grad_clip,
                    use_cuda=args.cuda, loss_weights=loss_weights)

  # Start training
  evaluator.evaluate(test_loader, step=0, tfLogger=eval_tfLogger, dataset=test_dataset)
  for epoch in range(start_epoch, args.epochs):
    scheduler.step(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    trainer.train(epoch, train_loader, optimizer, current_lr,
                  print_freq=args.print_freq,
                  train_tfLogger=train_tfLogger, 
                  is_debug=args.debug,
                  evaluator=evaluator, 
                  test_loader=test_loader, 
                  eval_tfLogger=eval_tfLogger,
                  test_dataset=test_dataset)

  # Final test
  print('Test with best model:')
  checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
  model.module.load_state_dict(checkpoint['state_dict'])
  evaluator.evaluate(test_loader, dataset=test_dataset)

  # Close the tensorboard logger
  train_tfLogger.close()
  eval_tfLogger.close()


if __name__ == '__main__':
  # parse the config
  args = get_args(sys.argv[1:])
  main(args)
