import os
import torch
import torch.nn.functional as F
from utils import Adder,class_indices,write_and_print_logs,get_class_indices
from dice_score import multiclass_iou
from data import My_dataset
from torch.utils.data import DataLoader
from visualizer import get_local

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from collections import OrderedDict
import cv2

from ptflops import get_model_complexity_info
from thop import profile
# from torchsummary import summary
from torchstat import stat

def visualize_grid_to_grid(args,index, attention_name, att_map, image, alpha=0.6):
    mask = Image.fromarray(att_map).resize((image.shape[1],image.shape[0]))
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    
    ax[0].imshow(image)
    ax[0].axis('off')
    
    ax[1].imshow(image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.savefig(os.path.join(args.attention_map_dir,'%s__%d.png'%(attention_name,index)))

def _eval(model, args):
    os.makedirs(args.attention_map_dir,exist_ok=True)
    # data_val = My_dataset(args.data_dir,datasets_type="val",datasets_name=args.datasets_name)
    data_val = My_dataset(args.data_dir,datasets_type="test",datasets_name=args.datasets_name)

    data_loader_val = DataLoader(data_val, batch_size=1, shuffle=False,num_workers=args.num_worker)

    if args.resume:
        state = torch.load(args.resume,map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state['model'].items():
            name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
            new_state_dict[name] = v #新字典的key值对应的value一一对应
        model.load_state_dict(new_state_dict)
        print('Load checkpoint.')

    model.eval()
    # macs,params=get_model_complexity_info(model,(3,180,240),as_strings=True,print_per_layer_stat=True, verbose=True)

    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # # inputs=
    # input = torch.randn(1, 3, 180,240)
    # flops, params = profile(model,inputs=(input.cuda(), ))
    # summary(model, input_size=(3, 224, 224), batch_size=-1)
    # input = torch.randn(3, 180,224)
    # stat(model.cpu(),(3, 180,224))

    # firing_rate_mean_adder = Adder()
    # TP_percent_mean_adder = Adder()
    # FP_2_TP_mean_adder = Adder()
    # iou_cls_adder = [Adder() for _ in range(args.num_classes)]
    # TP_percent_cls_adder = [Adder() for _ in range(args.num_classes)]
    # FP_2_TP_cls_adder = [Adder() for _ in range(args.num_classes)]
    # iou_mean_adder = Adder()
    iou=[]
    fire_rate=[]
    with torch.no_grad():
        for idx, data in enumerate(data_loader_val):
            input_img = data["img"]
            label_img = data["label"]
            input_img = input_img.to(dtype=torch.float32).cuda()
            label_img = label_img.to(dtype=torch.float32).cuda()
            label_img = label_img.squeeze(1)
            label_img = torch.tensor(label_img.cpu().numpy(),dtype=torch.long).cuda()
            label_img = F.one_hot(label_img, args.num_classes).permute(0, 3, 1, 2).float()

            pred1,firingrate = model(input_img)

            # firingrate=model.get_spike_mat()
            fire_rate.append(firingrate)
            # firing_rate_mean_adder()
            pred = pred1[2]
            pred = F.one_hot(pred.argmax(dim=1), args.num_classes).permute(0, 3, 1, 2).float()

            iou_score_list,TP_percent_list,FP_2_TP_list = multiclass_iou(pred, label_img, reduce_batch_first=True)
            iou_score_list=torch.mean(iou_score_list)

            # iou_temp_adder = Adder()
            # for index,i in enumerate(iou_score_list):
            #     iou_cls_adder[index](i)
            #     iou_temp_adder(i)
            # TP_percent_temp_adder = Adder()
            # for index,i in enumerate(TP_percent_list):
            #     TP_percent_cls_adder[index](i)
            #     TP_percent_temp_adder(i)
            # FP_2_TP_temp_adder = Adder()
            # for index,i in enumerate(FP_2_TP_list):
            #     FP_2_TP_cls_adder[index](i)
            #     FP_2_TP_temp_adder(i)
            iou.append(iou_score_list.numpy())
            # iou_mean_adder()
            # TP_percent_mean_adder(TP_percent_temp_adder.average())
            # FP_2_TP_mean_adder(FP_2_TP_temp_adder.average())

            cache = get_local.cache
            image = input_img.squeeze().permute(1,2,0).cpu().detach().numpy()

            img_path = data["img_path"][0]
            img_path = img_path.replace("events","images")
            img_path = img_path.replace("npy","png")
            
            # if iou_score_list.numpy()>=0.63:
            if img_path=='/sqy/CamVid/val/0016E5_07961.png' or img_path=='/sqy/CamVid/val/0016E5_08059.png' or img_path=='/sqy/CamVid/val/0016E5_08109.png':
                # rgb_image = cv2.imread(img_path)
                # rgb_image = cv2.resize(rgb_image,(240,180))
      
                # print(img_path,iou_score_list.numpy())
                # image = data["img"]
                # mask_true = data["label"]
                # image = image.to(dtype=torch.float32).cuda()
                # masks_pred = pred1
                # masks_pred = masks_pred[2]
                # masks_pred = masks_pred.argmax(dim=1)
                
                # img_name = os.path.basename(img_path)
                # tgt_img_name = os.path.join(args.attention_map_dir,"%s"%(img_name))
                # mask_true = mask_true[0]
                # mask_pred = masks_pred[0]
                # rgb_image = cv2.imread(img_path)
                # rgb_image = cv2.resize(rgb_image,(240,180))
                # mask_pred = mask_pred.squeeze().cpu().detach().numpy().astype(np.uint8)
                # mask_true = mask_true.squeeze().cpu().detach().numpy().astype(np.uint8)
                # # print(mask_true.shape)
                # # print(np.unique(mask_pred))
                # # print(np.unique(mask_true))
                # image = SegmentationMapsOnImage(mask_pred, shape=rgb_image.shape).draw_on_image(rgb_image)[0]
                # img_src = SegmentationMapsOnImage(mask_true, shape=rgb_image.shape).draw_on_image(rgb_image)[0]
                # image = np.hstack((img_src,image))
                # cv2.imwrite(tgt_img_name,image)

                attention_name = "mem_update.forward"
                for att_index in range(len(cache[attention_name])):
                    att_map = cache[attention_name][att_index][:,0,:,:,:]
                    att_map = np.mean(att_map,axis=0)
                    att_map = np.mean(att_map,axis=0)
                    visualize_grid_to_grid(args,att_index,img_path.split('.')[-2].split('/')[-1],att_map,image)

            # if idx >= 232:
            #     break
            # attention_name = "SpatialAttention.forward"
            # for att_index in range(len(cache[attention_name])):
            #     att_map = cache[attention_name][att_index][0,0,0,:,:]
            #     # visualize_grid_to_grid(args,att_index,attention_name,att_map,image)
            # attention_name = "TCSA.forward"
            # for att_index in range(len(cache[attention_name])):
            #     att_map = cache[attention_name][att_index][0,0,0,:,:]
            #     # visualize_grid_to_grid(args,att_index,attention_name,att_map,image)
            # attention_name = "mem_update.forward"
            # for att_index in range(len(cache[attention_name])):
            #     att_map = cache[attention_name][att_index][:,0,:,:,:]
            #     att_map = np.mean(att_map,axis=0)
            #     att_map = np.mean(att_map,axis=0)
            #     visualize_grid_to_grid(args,att_index,attention_name,att_map,image)

            # 
            #     break
        
            
            # for i in range(1):
                

    # for map in list(cache.keys()):
    #     print(map,len(cache[map]))
    #     for i in range(len(cache[map])):
    #         print(cache[map][i].shape)
    print("Eval: mean firing rate: %f, mean iou: %f"%(np.mean(fire_rate),np.mean(iou)))

    # class_indices = get_class_indices(args.datasets_name)
    # for key,value in class_indices.items():
    #     print('%s iou: %.3f TP_per: %.3f FP_2_TP: %.3f'%(\
    #         key,iou_cls_adder[value].average(),TP_percent_cls_adder[value].average(),FP_2_TP_cls_adder[value].average()))
