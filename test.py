import torch
import time
import numpy as np
import random
import os
import json
import glob
from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
# import util.eval as eval

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name,pretrained_name):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    print(pretrained_name)
    import stpls3d_inst
    dataset = stpls3d_inst.Dataset(test=True)
    dataset.testLoader()
    dataloader = dataset.test_data_loader

    with torch.no_grad():
        model = model.eval()

        total_end1 = 0.
        matches = {}
        for i, batch in enumerate(dataloader):

            # inference
            start1 = time.time()
            preds = model_fn(batch, model)
            end1 = time.time() - start1

            # decode results for evaluation
            N = batch['feats'].shape[0]
            test_scene_name = os.path.basename(dataset.test_file_names[int(batch['id'][0])].split('/')[-1]).strip('.pth')
            print (test_scene_name)
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
            matches[test_scene_name] = {}
            matches[test_scene_name]['seg_gt'] = batch['labels']
            matches[test_scene_name]['seg_pred'] = semantic_pred
            if False: #cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)
            

        # evaluate semantic segmantation accuracy and mIoU
        if  True: #cfg.split == 'val':
            seg_accuracy = evaluate_semantic_segmantation_accuracy(matches)
            logger.info("semantic_segmantation_accuracy: {:.4f}".format(seg_accuracy))
            iou_list = evaluate_semantic_segmantation_miou(matches)
            logger.info(iou_list)
            iou_list = torch.tensor(iou_list)
            miou = iou_list.mean()
            logger.info("semantic_segmantation_mIoU: {:.4f}".format(miou))
#             dict_hasil = {"semantic_segmantation_accuracy":seg_accuracy,"semantic_segmantation_IoU":iou_list,"semantic_segmantation_mIoU":miou}
            path_eval_pth= pretrained_name.split("/")
            path_eval_pth.insert(-1,"eval")
            path_eval_pth = "/".join(path_eval_pth)
            print(" Saving to ==========>",end=" ")
            print(path_eval_pth)
            torch.save((seg_accuracy,iou_list,miou),path_eval_pth)
#             with open(pretrained_name[:-4]+".json", 'w') as json_file:
#                 json.dump(dict_hasil, json_file)
            
def evaluate_semantic_segmantation_accuracy(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != -100] == seg_pred_all[seg_gt_all != -100]).sum()
    whole = (seg_gt_all != -100).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy

def evaluate_semantic_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = []
    for _index in seg_gt_all.unique():
        if _index != -100:
            intersection = ((seg_gt_all == _index) &  (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union
            iou_list.append(iou)

    return iou_list

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()
    torch.backends.cudnn.enabled=False

    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))
    
    for f_npm in sorted(glob.glob("/notebooks/pretrain/unetaspp/unetaspp2**.pth")):
        from model.seg_model import U_NET as Network
        from model.seg_model import model_fn_decorator


        model = Network(cfg)

        use_cuda = torch.cuda.is_available()
        logger.info('cuda available: {}'.format(use_cuda))
        assert use_cuda
        model = model.cuda()

        logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))
        model_fn = model_fn_decorator(test=True)

        # load model
        utils.checkpoint_restore(cfg, model, None, cfg.exp_path, cfg.config.split('/')[-1][:-5], 
            use_cuda, cfg.test_epoch, dist=False, f=f_npm)      
        # resume from the latest epoch, or specify the epoch to restore
    #     pretrain/unet/unet2-000000300.pth .split("/")[-1][:-4]

        # evaluate
        test(model, model_fn, data_name, f_npm)
        
    for i,val in enumerate(sorted(glob.glob("/notebooks/pretrain/unetaspp/eval/unetaspp2-**.pth"))):
        seg_accuracy,iou_list,miou = torch.load(val)
        if i ==0:
            n_seg_accuracy = seg_accuracy.cpu()
            n_iou_list = np.array(iou_list.cpu())
            n_miou = miou.cpu()      
        else:
            n_seg_accuracy = np.vstack((n_seg_accuracy,seg_accuracy.cpu()))
            n_iou_list = np.vstack((n_iou_list,np.array(iou_list.cpu())))
            n_miou = np.vstack((n_miou,miou.cpu()))
    torch.save((n_seg_accuracy,n_iou_list,n_miou),"/notebooks/total_eval_unetaspp.pth")