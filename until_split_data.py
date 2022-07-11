# from HAIS.util.config import cfg
import yaml
import torch

def load_yaml(folder_name = 'HAIS/config/hais_run_stpls3d.yaml'):
    with open(folder_name) as f:
        my_dict = yaml.safe_load(f)
    return my_dict

def save_yaml(data,folder_name = 'HAIS/config/hais_run_stpls3d.yaml'):
    with open('HAIS/config/hais_run_stpls3d.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def insert_word(fn,idx=-1,name="train"):
    fn= fn.split("/")
    fn.insert(-1,name)
    fn = "/".join(fn)
    return fn

files_train= glob.glob("/storage/Synthetic_v3_InstanceSegmentation/train_seg_50x25/**.pth")
train_size = int(0.8 * len(files_train))
test_size = len(files_train) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(files_train, [train_size, test_size])
os.makedirs('/storage/Synthetic_v3_InstanceSegmentation/train_seg_50x25/train/',exist_ok=True)
os.makedirs('/storage/Synthetic_v3_InstanceSegmentation/train_seg_50x25/val/',exist_ok=True)
for f in files_train : 
    if f in train_dataset:
        r_f = insert_word(fn=f,name="train")
        os.rename(f, r_f)

    if f in val_dataset:
        r_f = insert_word(fn=f,name="val")
        os.rename(f, r_f)