import spconv
import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import functools
import sys
sys.path.append('../../')

from lib.hais_ops.functions import hais_ops
from util import utils

class U_NET(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        width = cfg.width
        classes = cfg.classes
        block_reps = cfg.block_reps

        self.point_aggr_radius = cfg.point_aggr_radius
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.score_mode = cfg.score_mode

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module
        

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if cfg.use_coords:
            input_c += 3

        self.cfg = cfg

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, width, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        if cfg.aspp == True:
            from model.unet_aspp import Unet_aspp
            self.unet = Unet_aspp([width, 2*width, 3*width, 4*width, 5*width, 6*width, 7*width], norm_fn, block_reps)#, indice_key_id=1)
        else:
            from model.unet_base import unet_base
            self.unet = unet_base([width, 2*width, 3*width, 4*width, 5*width, 6*width, 7*width], norm_fn, block_reps)#, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(
            norm_fn(width),
            nn.ReLU()
        )

        # semantic segmentation branch
        self.semantic_linear = nn.Sequential(
            nn.Linear(width, width, bias=True),
            norm_fn(width),
            nn.ReLU(),
            nn.Linear(width, classes)
        )
        self.apply(self.set_bn_init)


        # fix module
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic_linear': self.semantic_linear}
        
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        # load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, input_map):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        # semantic segmentation
        semantic_scores = self.semantic_linear(output_feats)   # (N, nClass), float
        ret['semantic_scores'] = semantic_scores

        return ret
    
def model_fn_decorator(test=False):
    # config
    from util.config import cfg

    class_weight = torch.FloatTensor(cfg.class_weight).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label,weight=class_weight).cuda()
    if cfg.diceloss :
        from diceloss import DiceLoss,make_one_hot
        criterion = DiceLoss(weight=class_weight,ignore_index=cfg.ignore_label).cuda()
    else :
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label,weight=class_weight).cuda()
    
    def test_model_fn(batch, model):
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()              # (N, C), float32, cuda
        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = hais_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map)
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
      
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores        

        return preds
        
    def model_fn(batch, model):
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        if cfg.diceloss:
            labels = make_one_hot(torch.reshape(batch['labels'].cpu(),(len(batch['labels'] ),1)),15)               # (N), long, cuda
            labels = labels.cuda()
        else :
            labels = batch['labels'].cuda()
        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = hais_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map)
        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
       

        loss_inp = {}

        loss_inp['semantic_scores'] = (semantic_scores, labels)
       
        loss, loss_out = loss_fn(loss_inp)

        # accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
           
            visual_dict = {}
            visual_dict['loss'] = loss
           

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp):

        loss_out = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        
        semantic_loss = criterion(semantic_scores, semantic_labels)

        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        loss = cfg.loss_weight[0] * semantic_loss #+ cfg.loss_weight[1] * offset_norm_loss


        return loss, loss_out


    if test:
        fn = test_model_fn
    else:
        fn = model_fn

    return fn
