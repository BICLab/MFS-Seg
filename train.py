from email.policy import strict
import os
import torch
from data import My_dataset
from torch.utils.data import DataLoader
from utils import Adder, Timer, check_lr,draw,write_and_print_logs
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import numpy as np
from apex import amp
from spikingjelly.activation_based import functional
from thop import profile
from torch.utils.data.distributed import DistributedSampler
from torchstat import stat


def _train(model, args):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    data_train = My_dataset(args.data_dir,datasets_type="train",datasets_name=args.datasets_name)
    data_loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker)
    data_val = My_dataset(args.data_dir,datasets_type="test",datasets_name=args.datasets_name)
    data_loader_val = DataLoader(data_val, batch_size=1, shuffle=False,num_workers=args.num_worker)

    max_iter = len(data_loader_train)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume,map_location='cuda:0')
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

        model.load_state_dict(state['model'])
        write_and_print_logs(args.text_logs_dir,'Resume from %d'%epoch)
        epoch += 1
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1") #
    model = torch.nn.DataParallel(model)

    writer = SummaryWriter(args.logs_save_dir)
    epoch_pixel_adder = Adder()
    iter_pixel_adder = Adder()
    epoch_timer = Timer('s')
    iter_timer = Timer('s')
    best_iou = -1
    img_name_index = 0
    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(data_loader_train):

            input_img = batch_data["img"]
            label_img = batch_data["label"]
            if isinstance(input_img,list):
                for i in range(len(input_img)):
                    input_img[i] = input_img[i].to(dtype=torch.float32).cuda()
            else:
                input_img = input_img.to(dtype=torch.float32).cuda()
            label_img = label_img.to(dtype=torch.float32).cuda()

            optimizer.zero_grad()
            pred_img,fire = model(input_img)

            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')

            label_img = label_img.squeeze(1)#去掉第一维度
            label_img2 = label_img2.squeeze(1)
            label_img4 = label_img4.squeeze(1)

            label_img = torch.tensor(label_img.cpu().numpy(),dtype=torch.long).cuda()
            label_img2 = torch.tensor(label_img2.cpu().numpy(),dtype=torch.long).cuda()
            label_img4 = torch.tensor(label_img4.cpu().numpy(),dtype=torch.long).cuda()

            # label_img_onehot = F.one_hot(label_img, args.num_classes).permute(0, 3, 1, 2).float()
            # label_img2_onehot = F.one_hot(label_img2, args.num_classes).permute(0, 3, 1, 2).float()
            # label_img4_onehot = F.one_hot(label_img4, args.num_classes).permute(0, 3, 1, 2).float()
            # label_img_onehot = F.one_hot(label_img, args.num_classes).permute(0, 3, 1, 2)
            # label_img2_onehot = F.one_hot(label_img2, args.num_classes).permute(0, 3, 1, 2)
            # label_img4_onehot = F.one_hot(label_img4, args.num_classes).permute(0, 3, 1, 2)

            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)

            loss_content = l1 * args.min_scale_loss + l2 * args.mid_scale_loss + l3 * args.max_scale_loss
            # loss_content=l3
            
            with amp.scale_loss(loss_content, optimizer) as scaled_loss:
                scaled_loss.backward()

            iter_pixel_adder(loss_content.item())
            epoch_pixel_adder(loss_content.item())

            optimizer.step()
            functional.reset_net(model)

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                write_and_print_logs(args.text_logs_dir,"Time: %7.4f Epoch: %03d Iter: %4d/%4d Loss content: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, iter_pixel_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
            del l1,l2,l3
            del loss_content
        torch.cuda.empty_cache()

        if epoch_idx % args.save_freq == 0 :
            save_name = os.path.join(args.model_save_dir, 'lastmodel.pkl')
            # save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)

            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)

        write_and_print_logs(args.text_logs_dir,"ep:%d time:%.2f Epoch Pixel Loss:%.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average()))
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            current_iou = _valid(model, args, epoch_idx, data_loader_val)
            write_and_print_logs(args.text_logs_dir,'%03d epoch Average Current IOU %.4f Best IOU %.4f' % (epoch_idx, current_iou, best_iou))
            writer.add_scalar('val IOU', current_iou, epoch_idx)
            if current_iou >= best_iou:
                best_iou = current_iou
                if epoch_idx>70:
                    save_name = os.path.join(args.model_save_dir, 'epoch_%d_iou_%.3f.pkl' % (epoch_idx,best_iou))
                    torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch_idx}, save_name)
                
                
                # with torch.no_grad():
                #     model.eval()
                #     for img_index,data in enumerate(data_loader_val):
                #         image = data["img"]
                #         mask = data["label"]
                #         image = image.to(dtype=torch.float32).cuda()
                #         masks_pred = model(image)
                #         masks_pred = masks_pred[2]
                #         masks_pred = masks_pred.argmax(dim=1)
                #         # for i in range(image.shape[0]):
                #         for i in range(1):
                #             os.makedirs(os.path.join(args.result_dir,"epoch_%d_iou_%.3f"%(epoch_idx,best_iou)),exist_ok=True)
                #             img_name = os.path.join(args.result_dir,"epoch_%d_iou_%.3f"%(epoch_idx,best_iou),"%d_val.jpg"%(img_name_index))
                #             img_name_index += 1
                #             draw(image[i],mask[i].squeeze(0),masks_pred[i],img_name)
                torch.cuda.empty_cache()
                model.train()

from email.policy import strict
import os
import torch
from data import My_dataset
from torch.utils.data import DataLoader
from utils import Adder, Timer, check_lr,draw,write_and_print_logs
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import numpy as np
from apex import amp
from spikingjelly.activation_based import functional
from thop import profile
from torch.utils.data.distributed import DistributedSampler
from torchstat import stat


def _train(model, args):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    data_train = My_dataset(args.data_dir,datasets_type="train",datasets_name=args.datasets_name)
    data_loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker)
    data_val = My_dataset(args.data_dir,datasets_type="test",datasets_name=args.datasets_name)
    data_loader_val = DataLoader(data_val, batch_size=1, shuffle=False,num_workers=args.num_worker)

    max_iter = len(data_loader_train)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume,map_location='cuda:0')
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

        model.load_state_dict(state['model'])
        write_and_print_logs(args.text_logs_dir,'Resume from %d'%epoch)
        epoch += 1
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1") #
    model = torch.nn.DataParallel(model)

    writer = SummaryWriter(args.logs_save_dir)
    epoch_pixel_adder = Adder()
    iter_pixel_adder = Adder()
    epoch_timer = Timer('s')
    iter_timer = Timer('s')
    best_iou = -1
    img_name_index = 0
    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(data_loader_train):

            input_img = batch_data["img"]
            label_img = batch_data["label"]
            if isinstance(input_img,list):
                for i in range(len(input_img)):
                    input_img[i] = input_img[i].to(dtype=torch.float32).cuda()
            else:
                input_img = input_img.to(dtype=torch.float32).cuda()
            label_img = label_img.to(dtype=torch.float32).cuda()

            optimizer.zero_grad()
            pred_img,fire = model(input_img)

            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')

            label_img = label_img.squeeze(1)#去掉第一维度
            label_img2 = label_img2.squeeze(1)
            label_img4 = label_img4.squeeze(1)

            label_img = torch.tensor(label_img.cpu().numpy(),dtype=torch.long).cuda()
            label_img2 = torch.tensor(label_img2.cpu().numpy(),dtype=torch.long).cuda()
            label_img4 = torch.tensor(label_img4.cpu().numpy(),dtype=torch.long).cuda()

            # label_img_onehot = F.one_hot(label_img, args.num_classes).permute(0, 3, 1, 2).float()
            # label_img2_onehot = F.one_hot(label_img2, args.num_classes).permute(0, 3, 1, 2).float()
            # label_img4_onehot = F.one_hot(label_img4, args.num_classes).permute(0, 3, 1, 2).float()
            # label_img_onehot = F.one_hot(label_img, args.num_classes).permute(0, 3, 1, 2)
            # label_img2_onehot = F.one_hot(label_img2, args.num_classes).permute(0, 3, 1, 2)
            # label_img4_onehot = F.one_hot(label_img4, args.num_classes).permute(0, 3, 1, 2)

            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)

            loss_content = l1 * args.min_scale_loss + l2 * args.mid_scale_loss + l3 * args.max_scale_loss
            # loss_content=l3
            
            with amp.scale_loss(loss_content, optimizer) as scaled_loss:
                scaled_loss.backward()

            iter_pixel_adder(loss_content.item())
            epoch_pixel_adder(loss_content.item())

            optimizer.step()
            functional.reset_net(model)

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                write_and_print_logs(args.text_logs_dir,"Time: %7.4f Epoch: %03d Iter: %4d/%4d Loss content: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, iter_pixel_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
            del l1,l2,l3
            del loss_content
        torch.cuda.empty_cache()

        if epoch_idx % args.save_freq == 0 :
            save_name = os.path.join(args.model_save_dir, 'lastmodel.pkl')
            # save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)

            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)

        write_and_print_logs(args.text_logs_dir,"ep:%d time:%.2f Epoch Pixel Loss:%.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average()))
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            current_iou = _valid(model, args, epoch_idx, data_loader_val)
            write_and_print_logs(args.text_logs_dir,'%03d epoch Average Current IOU %.4f Best IOU %.4f' % (epoch_idx, current_iou, best_iou))
            writer.add_scalar('val IOU', current_iou, epoch_idx)
            if current_iou >= best_iou:
                best_iou = current_iou
                if epoch_idx>70:
                    save_name = os.path.join(args.model_save_dir, 'epoch_%d_iou_%.3f.pkl' % (epoch_idx,best_iou))
                    torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch_idx}, save_name)
                
                
                # with torch.no_grad():
                #     model.eval()
                #     for img_index,data in enumerate(data_loader_val):
                #         image = data["img"]
                #         mask = data["label"]
                #         image = image.to(dtype=torch.float32).cuda()
                #         masks_pred = model(image)
                #         masks_pred = masks_pred[2]
                #         masks_pred = masks_pred.argmax(dim=1)
                #         # for i in range(image.shape[0]):
                #         for i in range(1):
                #             os.makedirs(os.path.join(args.result_dir,"epoch_%d_iou_%.3f"%(epoch_idx,best_iou)),exist_ok=True)
                #             img_name = os.path.join(args.result_dir,"epoch_%d_iou_%.3f"%(epoch_idx,best_iou),"%d_val.jpg"%(img_name_index))
                #             img_name_index += 1
                #             draw(image[i],mask[i].squeeze(0),masks_pred[i],img_name)
                torch.cuda.empty_cache()
                model.train()

