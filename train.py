import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import datasets
import settings
from model import  ImgNet_V1, TxtNet_V1, ImgNet, TxtNet
from metric import compress_wiki, compress_nus,compress, calculate_top_map, load_feature_construct_H, generate_G_from_H,calculate_map
import os.path as osp
import random
import numpy as np
import copy
from tools import build_G_from_S, generate_robust_S
import csv
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter('HCAC/log')
#返回 归一化后 的数据集图文特征，其中图像特征为基于预训练模型提取的特征，文本特征为数据集自带的特征
def extract_features(img_model, dataloader, feature_loader):
    """
    Extract features.
    """
    if settings.DATASET == "WIKI":
        sample_num = len(feature_loader.dataset.label)
    else:
        sample_num = feature_loader.dataset.train_labels.shape[0]
    img_model.cuda().eval()
    img_features = torch.zeros(sample_num, 4096).cuda()
    if settings.DATASET == "MIRFlickr":
        txt_features = torch.zeros(sample_num, 1386).cuda()
    if settings.DATASET == "WIKI":
        txt_features = torch.zeros(sample_num, 10).cuda()
    if settings.DATASET == "NUSWIDE":
        txt_features = torch.zeros(sample_num, 1000).cuda()
    if settings.DATASET == "MSCOCO":
        txt_features = torch.zeros(sample_num, 2000).cuda()
    with torch.no_grad():
        for i, (img, F_T, _, index) in enumerate(dataloader):
            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            img_features[index, :], _, _ = img_model(img)
            txt_features[index, :] = F_T

    return F.normalize(img_features), F.normalize(txt_features)




if settings.DATASET == "WIKI":
    train_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_train_transform)
    test_dataset = datasets.WIKI(root=settings.DATA_DIR, train=False, transform=datasets.wiki_test_transform)
    database_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_test_transform)
    feature_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_test_transform)
    feature_dataset_test = datasets.WIKI(root=settings.DATA_DIR, train=False, transform=datasets.wiki_test_transform)
    feature_dataset_database = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_test_transform)
if settings.DATASET == "MIRFlickr":
    train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
    test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
    database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)
    feature_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_test_transform)
    feature_dataset_test = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
    feature_dataset_database = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

if settings.DATASET == "NUSWIDE":
    train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
    test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
    database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)
    feature_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_test_transform)
    feature_dataset_test = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
    feature_dataset_database = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

if settings.DATASET == "MSCOCO":
    train_dataset = datasets.MSCOCO(train=True, transform=datasets.coco_train_transform)
    test_dataset = datasets.MSCOCO(train=False, database=False, transform=datasets.coco_test_transform)
    database_dataset = datasets.MSCOCO(train=False, database=True, transform=datasets.coco_test_transform)
    feature_dataset = datasets.MSCOCO(train=True, transform=datasets.coco_test_transform)
    feature_dataset_test = datasets.MSCOCO(train=False, database=False, transform=datasets.coco_test_transform)
    feature_dataset_database = datasets.MSCOCO(train=False, database=True, transform=datasets.coco_test_transform)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=settings.NUM_WORKERS,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=settings.NUM_WORKERS)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=settings.NUM_WORKERS)

feature_loader = torch.utils.data.DataLoader(dataset=feature_dataset,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=settings.NUM_WORKERS)
feature_loader_test = torch.utils.data.DataLoader(dataset=feature_dataset_test,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=settings.NUM_WORKERS)
feature_loader_database = torch.utils.data.DataLoader(dataset=feature_dataset_database,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=settings.NUM_WORKERS)



FeatNet_I = ImgNet_V1(code_len=settings.CODE_LEN)
if settings.DATASET == "WIKI":
    sample_num = len(feature_loader.dataset.label)
else:
    sample_num = feature_loader.dataset.train_labels.shape[0]

global_imgs, global_txts = extract_features(FeatNet_I, feature_loader, feature_loader)



feature_img = global_imgs
feature_txt = global_txts
dim = sample_num

F_I = feature_img
S_I = F_I.mm(F_I.t())
F_T = feature_txt
S_T = F_T.mm(F_T.t())

settings.logger.info('dataset %s, nnk %.4f, nno %.4f, total epoch %d, eval interval %d, sim1 %.4f' % (settings.DATASET, settings.nnk, nno,  settings.NUM_EPOCH, settings.EVAL_INTERVAL,settings.sim1))


S_high_crs = F.normalize(S_I).mm(F.normalize(S_T).t())
if settings.DATASET == "MIRFlickr" or settings.DATASET == "MSCOCO":
    sim1 = 0.5 * S_I + 0.1 * S_T + 0.4 * (S_high_crs + S_high_crs.t()) / 2
    sim1 = sim1  * 1.4
else:
    sim1 = 0.4 * S_I + 0.3 * S_T + 0.3 * (S_high_crs + S_high_crs.t()) / 2
    sim1 = sim1  * 1.4

final_sim = sim1

# fianl_sim = generate_robust_S(final_sim, settings.alpha, settings.beta)

if settings.DATASET == "WIKI":
    sample_num = len(feature_loader_test.dataset.label)
else:
    sample_num = feature_loader_test.dataset.train_labels.shape[0]

global_imgs_test, global_txts_test = extract_features(FeatNet_I, feature_loader_test, feature_loader_test)


feature_img_test = global_imgs_test
feature_txt_test = global_txts_test

F_I_test = feature_img_test
S_I_test = F_I_test.mm(F_I_test.t())
F_T_test = feature_txt_test
S_T_test = F_T_test.mm(F_T_test.t())

S_high_crs_test = F.normalize(S_I_test).mm(F.normalize(S_T_test).t())
if settings.DATASET == "MIRFlickr" or settings.DATASET == "MSCOCO":

    sim1_test = 0.5 * S_I_test + 0.1 * S_T_test + 0.4 * (S_high_crs_test + S_high_crs_test.t()) / 2
    sim1_test = sim1_test  * 1.4
else:
    sim1_test = 0.4 * S_I_test + 0.3 * S_T_test + 0.3 * (S_high_crs_test + S_high_crs_test.t()) / 2
    sim1_test = sim1_test  * 1.4

final_sim_test = sim1_test


#################### 数据库样本的G构建 ##############

# NUSWIDE 时不使用全局数据库样本
if settings.DATASET != 'NUSWIDE' and settings.DATASET != 'MSCOCO':
        if settings.DATASET == "WIKI":
            sample_num = len(feature_loader_test.dataset.label)
        else:
            sample_num = feature_loader_database.dataset.train_labels.shape[0]

        global_imgs_database, global_txts_database = extract_features(FeatNet_I, feature_loader_database, feature_loader_database)


        feature_img_database = global_imgs_database
        feature_txt_database = global_txts_database

        F_I_database = feature_img_database
        S_I_database = F_I_database.mm(F_I_database.t())
        F_T_database = feature_txt_database
        S_T_database = F_T_database.mm(F_T_database.t())

        S_high_crs_database = F.normalize(S_I_database).mm(F.normalize(S_T_database).t())
        if settings.DATASET == "MIRFlickr":
            sim1_database = 0.5 * S_I_database + 0.1 * S_T_database + 0.4 * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = sim1_database  * 1.4
        elif settings.DATASET == "NUSWIDE":
            sim1_database = 0.4 * S_I_database + 0.3 * S_T_database + 0.3 * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = sim1_database  * 1.4
        else:
            sim1 = 0.5 * S_I + 0.1 * S_T + 0.4 * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = sim1 *  1.4
        final_sim_database = sim1_database
else:
        final_sim_database = 0



class Session:
    def __init__(self):
        self.logger = settings.logger
        torch.cuda.set_device(settings.GPU_ID)
        
        
        self.CodeNet_I = ImgNet_V1(code_len=settings.CODE_LEN)
        self.FeatNet_I = ImgNet_V1(code_len=settings.CODE_LEN)

        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_T = TxtNet_V1(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        if settings.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE" or settings.DATASET == "MSCOCO":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
          
        self.final_sim = 2 * final_sim - 1.0
        self.final_sim = generate_robust_S(self.final_sim, settings.alpha, settings.beta)


    def train(self, epoch):
        self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        
        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, txt, labels, batch_ind) in enumerate(train_loader):
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            S = self.final_sim[batch_ind, :]
            S = S[:, batch_ind]
            S = S.cuda()

            G = build_G_from_S(S, settings.K)
            batch_size = img.size(0)
            
            
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            _, hid_I, code_I = self.CodeNet_I(img, G)
            _, hid_T, code_T = self.CodeNet_T(txt, G)



            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)
            
            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())
            BT_BI = B_T.mm(B_I.t())

            
            loss1 = F.mse_loss(BI_BI, S)
            loss2 = (F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S)) * 0.5
            loss3 = F.mse_loss(BT_BT, S)
            diagonal = BI_BT.diagonal()
            all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
            loss4 = F.mse_loss(diagonal, 1.5 * all_1)

            loss = settings.LAMBDA1 * loss1 + 1 * loss2  + settings.LAMBDA2 * loss3 + settings.l4 * loss4 
            # writer.add_scalar('loss', loss, epoch * len(train_dataset) // settings.BATCH_SIZE + idx)
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Loss4: %.4f Total Loss: %.4f'
                    % (epoch + 1, settings.NUM_EPOCH, idx + 1, len(train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(), loss3.item(), loss4.item() ,loss.item()))

    def eval(self,avgScore):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        self.CodeNet_I.eval().cuda()  
        self.CodeNet_T.eval().cuda()


        if settings.DATASET == "MIRFlickr" or settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(database_loader, test_loader, self.CodeNet_I, self.CodeNet_T, database_dataset, test_dataset, final_sim_database, final_sim_test)
        if settings.DATASET == "NUSWIDE" or settings.DATASET == "MSCOCO":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_nus(database_loader, test_loader, self.CodeNet_I, self.CodeNet_T, database_dataset, test_dataset, final_sim_test)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        avgScore[0] = (avgScore[0] * avgScore[2] + MAP_I2T)/(avgScore[2]+1)
        avgScore[1] = (avgScore[1] * avgScore[2] + MAP_T2I)/(avgScore[2]+1)
        avgScore[2] += 1
        if MAP_I2T + MAP_T2I >= avgScore[3] + avgScore[4]:
            avgScore[3] = MAP_I2T
            avgScore[4] = MAP_T2I

        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f   avgI2T: %.4f avgT2I: %.4f bestPair:(%.3f,%.3f) evalNum:%d' % (MAP_I2T, MAP_T2I,avgScore[0],avgScore[1],avgScore[3],avgScore[4],avgScore[2]))
        self.logger.info('--------------------------------------------------------------------')


    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    
    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])



def main():
    # for random_id in range(30,50,1):
        random_id = 30
        torch.manual_seed(random_id)
        torch.cuda.manual_seed_all(random_id)
        settings.logger.info('random seed id: %d'%random_id)
        settings.logger.info('%.4f loss1, 1 loss2, %.4f loss3, %.4f loss4, %d bit, map@all!!!' % (settings.LAMBDA1, settings.LAMBDA2, settings.l4, settings.CODE_LEN))
        sess = Session()
        

        avgScore = [0.0,0.0,0,0,0]

        if settings.EVAL == True:
            sess.load_checkpoints()
            sess.eval(avgScore)

        else :
            for epoch in range(settings.NUM_EPOCH):
                # train the Model
                sess.train(epoch)
                # eval the Model
                if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                    sess.eval(avgScore)
                # save the model
                if epoch + 1 == settings.NUM_EPOCH:
                    # sess.save_checkpoints(step=epoch+1)
                    pass






if __name__ == '__main__':

        main()

   