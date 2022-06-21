import os
import csv
import logging
import math
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import json
from collections import OrderedDict
from glob import glob
from pathlib import Path

import numpy as np
import wandb
import yaml

_LOGGER = None

class ExpHandler:
    _home = Path.home()

    def __init__(self, en_wandb=False, args=None):
        project_name = os.getenv('WANDB_PROJECT', default='default_project')
        exp_name = os.getenv('exp_name', default='default_group')
        run_name = os.getenv('run_name', default='default_name')
        self._exp_id = f'{self._get_exp_id()}_{run_name}'
        self._exp_name = exp_name

        self._save_dir = os.path.join('{}/.exp/{}'.format(self._home, os.getenv('WANDB_PROJECT', default='default_project')),
                                    exp_name, self._exp_id)
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        if args.en_wandb:
            self.wandb_run = wandb.init(project=project_name, group=exp_name, name=run_name,settings=wandb.Settings(start_method="fork"))

        sym_dest = self._get_sym_path('N')
        os.symlink(self._save_dir, sym_dest)

        self._logger = self._init_logger()
        self._en_wandb = en_wandb


    @staticmethod
    def resume_sanity(args, old_conf):
        print('-' * 10, 'Resume sanity check', '-' * 10)
        old_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in old_conf.items()
                                if k not in args.resume_check_exclude_keys}
        new_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in vars(args).items()
                                if k not in args.resume_check_exclude_keys}
        print(f'Diff config: {set(old_config_hashable.items()) ^ set(new_config_hashable.items())}')
        assert old_config_hashable == new_config_hashable, 'Resume sanity check failed'

    def _get_sym_path(self, state):
        sym_dir = f'{self._home}/.exp/syms'
        if not os.path.exists(sym_dir):
            os.makedirs(sym_dir)

        sym_dest = os.path.join(sym_dir, '--'.join([self._exp_id, state, self._exp_name]))
        return sym_dest

    @property
    def save_dir(self):
        return self._save_dir

    @staticmethod
    def _get_exp_id():
        if not os.path.exists(f'{ExpHandler._home}/.core'):
            os.makedirs(f'{ExpHandler._home}/.core')
            core = Path(f'{ExpHandler._home}/.core/counter')
            core.touch(exist_ok=True)
            counter = Path(f'{ExpHandler._home}/.core/identifier').touch(exist_ok=True)
            counter.touch(exist_ok=True)
            f1 = open(core)
            f1.write('0')
            f2 = open(counter)
            f2.write("Y")
        with open(f'{ExpHandler._home}/.core/counter', 'r+') as f:
            counter = eval(f.read())
            f.seek(0)
            f.write(str(counter + 1))
        with open(f'{ExpHandler._home}/.core/identifier', 'r+') as f:
            identifier = f.read()[0]
        exp_id = '{}{:04d}'.format(identifier, counter)
        return exp_id

    def _init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(self._save_dir, f'{self._exp_id}_log.txt'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def save_config(self, args):
        conf = vars(args)
        conf['exp_id'] = self._exp_id
        conf['commit'] = os.getenv('commit', default='not_set')
        conf['run_id'] = self._exp_id.split('_')[0]
        if hasattr(self, 'wandb_run'):
            conf['wandb_id'] = self.wandb_run.id
        with open(f'{self._save_dir}/config.yaml', 'w') as f:
            yaml.dump(conf, f)

        if self._en_wandb:
            wandb.config.update(conf,allow_val_change=True)

    def write(self, prefix, eval_metrics=None, train_metrics=None, **kwargs):
        rowd = OrderedDict([(f'{prefix}/{k}', v) for k, v in kwargs.items() ])
        if eval_metrics:
            rowd.update([(f'{prefix}/eval_' + k, v) for k, v in eval_metrics.items()])
        if train_metrics:
            rowd.update([(f'{prefix}/train_' + k, v) for k, v in train_metrics.items()])

        path = os.path.join(self._save_dir, f'{self._exp_id}_{prefix}_summary.csv')
        initial = not os.path.exists(path)
        with open(path, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            if initial:
                dw.writeheader()
            dw.writerow(rowd)

        if self._en_wandb:
            wandb.log(rowd)

    def log(self, msg):
        self._logger.info(msg)

    def finish(self):
        Path(f'{self._save_dir}/finished').touch()
        os.rename(self._get_sym_path('N'), self._get_sym_path('Y'))

def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def get_env_variable(variables, default=None):
    for candidate in variables:
        if candidate in os.environ:
            return os.environ[candidate]
    return default

def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=" * 105,
    ]

    row_format = "{name:<60} {shape:>27} ={total_size:>15,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 105)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def create_logger(log_file, level=logging.INFO):
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    l = logging.getLogger('global')
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    l.propagate = False
    _LOGGER = l
    return l

def createJs(pred:dict,ori_path, out_path):
    js = parseJson(ori_path)
    for pic in pred.keys():
        comName, idx = pic.split('_')[0], int(pic.split('_')[-1])
        if pred[pic] in js[comName]['optional_tags']:
            fileName = pic+".jpg"
            if js[comName]['imgs_tags'][idx][fileName] is None:
                js[comName]['imgs_tags'][idx][fileName] = pred[pic]
            else:
                raise ValueError("Duplicated Prediction!")
        else:
            raise ValueError("Illegal Prediction!")
    # assert every prediction is set
    for com in js.keys():
        for tag in js[com]['imgs_tags']:
            assert tag.values() is not None, "Incomplete Prediction!"
    with open(out_path, 'w+', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False)
        print("Json Saved")

def parseJson(file):
    with open(str(file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 清理掉多余的空行
        raw = [line.strip('\n').strip(' ') for line in lines]
        raw_str = "".join(str(i) for i in raw)
        data = json.loads(raw_str)
        return data

def get_logger():
    return _LOGGER

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter(object):
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history, self.history_num = [], []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        assert num > 0
        if self.length > 0:
            self.history.append(val * num)
            self.history_num.append(num)
            if len(self.history) > self.length:
                del self.history[0]
                del self.history_num[0]

            self.val = val
            self.avg = np.sum(self.history) / np.sum(self.history_num)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def init_distributed(local_rank, args):
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['MASTER_ADDR'] = str(args.master_addr)

    world_size = n_nodes * args.nproc_per_node
    rank = node_id * args.nproc_per_node + local_rank
    dist.init_process_group(backend=args.backend, init_method='env://', world_size=world_size, rank=rank)
    print('[rank {:04d}]: distributed init: world_size={}, local_rank={}'.format(rank, world_size, local_rank), flush=True)
    
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank%num_gpus)
    
    return rank, world_size
