import argparse
import yaml
import os

def get_parser():
    # parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    # parser.add_argument('--config', type=str, default="hais_run_stpls3d.yaml",help='path to config file')

    # # pretrain
    # parser.add_argument('--pretrain', type=str, help='path to pretrain model')

    # parser.add_argument('--save_dir', type=str, default='exp', help='path to save model')

    # parser.add_argument('--dist', action='store_true', default=False, help='dist train')

    # parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')


    # args_cfg = parser.parse_args()
    class args_cfg:
        config = "config/unet.yaml"
        save_dir = "save_dir"
        pretrain = "exp"
        dist = False
        local_rank = 0
        
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join(cfg.save_dir, cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))