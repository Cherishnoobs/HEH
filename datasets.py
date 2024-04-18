import torch
import cv2
import scipy.io as scio
from PIL import Image
import settings
import numpy as np
import scipy.io as scio
from torchvision import transforms
import h5py
np.random.seed(6)

if settings.DATASET == "WIKI":

    label_set = scio.loadmat(settings.LABEL_DIR)    #加载raw_feature
    test_txt = np.array(label_set['T_te'], dtype=np.float)
    train_txt = np.array(label_set['T_tr'], dtype=np.float)

#构建训练集和测试集的 label（起始值归零处理）  和   图片（获取图片的路径）
    test_label = []
    with open(settings.TEST_LABEL, 'r') as f:
        for line in f.readlines(): 
            test_label.extend([int(line.split()[-1])-1])  #-1是为了将类别下标的起始值归为0

    test_img_name = []
    with open(settings.TEST_LABEL, 'r') as f:
        for line in f.readlines(): 
            test_img_name.extend([line.split()[1]])
            

    train_label = []
    with open(settings.TRAIN_LABEL, 'r') as f:
        for line in f.readlines(): 
            train_label.extend([int(line.split()[-1])-1])

    train_img_name = []
    with open(settings.TRAIN_LABEL, 'r') as f:
        for line in f.readlines(): 
            train_img_name.extend([line.split()[1]])

    wiki_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    wiki_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_feat_len = train_txt.shape[1] #长度为10

#构建WIKI类的目的-通过WIKI[index]操作，能够获取训练集或者测试集中的对应下标处的实例（图像tensor，文本feature，label标号），
    class WIKI(torch.utils.data.Dataset):

        def __init__(self, root, transform=None, target_transform=None, train=True):
            self.root = root #image的外层文件夹名
            self.transform = transform
            self.target_transform = target_transform
            self.f_name =['art','biology','geography','history','literature','media','music','royalty','sport','warfare']

            #要求：三个数组的同下标元素相互对应即可
            if train:
                self.label = train_label
                self.img_name = train_img_name
                self.txt = train_txt
            else:
                self.label = test_label
                self.img_name = test_img_name
                self.txt = test_txt

        def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            
            path = self.root + '/' + self.f_name[self.label[index]] + '/' + self.img_name[index] + '.jpg'
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            target = self.label[index]
            txt = self.txt[index]

            if self.transform is not None:
                img = self.transform(img)

            #train.py中并没有使用target_transform
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index #返回值为(图像tensor，文本feature，类别号，传入的index)

        def __len__(self):
            return len(self.label)


if settings.DATASET == "MIRFlickr":
    
    label_set = scio.loadmat(settings.LABEL_DIR)
    label_set = np.array(label_set['LAll'], dtype=np.float32)
    txt_set = scio.loadmat(settings.TXT_DIR)
    txt_set = np.array(txt_set['YAll'], dtype=np.float32)

    first = True
    #从24个不同的label中随机选取2000/5000个样本作为测试集和训练集（为何第一类是后续类别样本数量的两倍？）
    #测试集以外的样本全部作为数据库
    for label in range(label_set.shape[1]):
        index = np.where(label_set[:,label] == 1)[0]
        
        N = index.shape[0] #获取当前标签为1的数组的长度
        perm = np.random.permutation(N)
        index = index[perm]   #将index打乱
        
        if first:
            test_index = index[:160]
            train_index = index[160:160+400]
            first = False
        else:
            ind = np.array([i for i in list(index) if i not in (list(train_index)+list(test_index))])
            test_index = np.concatenate((test_index, ind[:80])) #加上上面的160个，一共是160+23*80 = 2000个
            train_index = np.concatenate((train_index, ind[80:80+200])) #加上上面的400个，一共是400+23*200 = 5000 均与论文中的数据描述对应


    database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])
    #如果训练集的样本数量不足5000，则从database中随机抽取差额数量的样本补足
    if train_index.shape[0] < 5000:
        pick = np.array([i for i in list(database_index) if i not in list(train_index)])
        N = pick.shape[0]
        perm = np.random.permutation(N)
        pick = pick[perm]
        res = 5000 - train_index.shape[0]
        train_index = np.concatenate((train_index, pick[:res]))


    indexTest = test_index
    indexDatabase = database_index
    indexTrain = train_index


    mir_train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    mir_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    txt_feat_len = txt_set.shape[1]
    #与WIKI类类似，只是读取的图片为数组而不是原始的图片，最后返回的值与WIKI相同
    class MIRFlickr(torch.utils.data.Dataset):
        def __init__(self, transform=None, target_transform=None, train=True, database=False):
            self.transform = transform
            self.target_transform = target_transform

            if train:
                self.train_labels = label_set[indexTrain]
                self.train_index = indexTrain
                self.txt = txt_set[indexTrain]
            elif database:
                self.train_labels = label_set[indexDatabase]
                self.train_index = indexDatabase
                self.txt = txt_set[indexDatabase]
            else:
                self.train_labels = label_set[indexTest]  
                self.train_index = indexTest
                self.txt = txt_set[indexTest]

        def __getitem__(self, index):

            mirflickr = h5py.File(settings.IMG_DIR, 'r', libver='latest', swmr=True)
            img, target = mirflickr['IAll'][self.train_index[index]], self.train_labels[index]
            img = Image.fromarray(np.transpose(img, (2, 1, 0)))
            mirflickr.close()
            
            txt = self.txt[index]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)

if settings.DATASET == "NUSWIDE":

    label_set = scio.loadmat(settings.LABEL_DIR)
    label_set = np.array(label_set['LAll'], dtype=np.float)
    txt_file = h5py.File(settings.TXT_DIR)
    txt_set = np.array(txt_file['YAll']).transpose()
    txt_file.close()


    first = True

    for label in range(label_set.shape[1]):
        index = np.where(label_set[:,label] == 1)[0]
        
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]
        
        if first:
            test_index = index[:200]
            train_index = index[200:700]
            first = False
        else:
            ind = np.array([i for i in list(index) if i not in (list(train_index)+list(test_index))])
            test_index = np.concatenate((test_index, ind[:200]))
            train_index = np.concatenate((train_index, ind[200:700]))

        
    database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])

    indexTest = test_index
    indexDatabase = database_index
    indexTrain = train_index


    nus_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    nus_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_feat_len = txt_set.shape[1]

#与MIRFlickr类基本类似
    class NUSWIDE(torch.utils.data.Dataset):

        def __init__(self, transform=None, target_transform=None, train=True, database=False):
            self.transform = transform
            self.target_transform = target_transform
            if train:
                self.train_labels = label_set[indexTrain]
                self.train_index = indexTrain
                self.txt = txt_set[indexTrain]
            elif database:
                self.train_labels = label_set[indexDatabase]
                self.train_index = indexDatabase
                self.txt = txt_set[indexDatabase]
            else:
                self.train_labels = label_set[indexTest]  
                self.train_index = indexTest
                self.txt = txt_set[indexTest]

        def __getitem__(self, index):

            nuswide = h5py.File(settings.IMG_DIR, 'r', libver='latest', swmr=True)
            #train_labels已经在__init__函数中进行了整理
            img, target = nuswide['IAll'][self.train_index[index]], self.train_labels[index]
            img = Image.fromarray(np.transpose(img, (2, 1, 0)))
            nuswide.close()
            
            txt = self.txt[index]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)

if settings.DATASET == "MSCOCO":
    # label_file = h5py.File(settings.LABEL_DIR)
    # label_set = np.array(label_file['LAll'], dtype=np.float32)
    # txt_file = h5py.File(settings.TXT_DIR)
    # txt_set = np.array(txt_file['YAll'], dtype=np.float)
    # label_file.close()
    # txt_file.close()
    

    # test_index = np.random.randint(low=0, high = 40137, size = 5000, dtype='l')
    # train_index = np.random.randint(low=40137, high = label_set.shape[0], size = 10000, dtype='l')
    # database_index = np.array([i for i in range(label_set.shape[0]) if i not in (list(test_index) )])
    # first = True

    # for label in range(label_set.shape[1]):
    #     index = np.where(label_set[:,label] == 1)[0]

    #     N = index.shape[0]
    #     perm = np.random.permutation(N)
    #     index = index[perm]

    #     if first:
    #         test_index = index[:62]
    #         train_index = index[62:62 + 125]
    #         first = False
    #     else:
    #         ind = np.array([i for i in list(index) if i not in (list(test_index) + list(train_index))])
    #         test_index = np.concatenate((test_index, ind[:62]))
    #         train_index = np.concatenate((train_index, ind[62:62 + 125]))

    # database_index = np.array([i for i in range(label_set.shape[0]) if i not in (list(test_index))])
    


    indexTest = np.arange(2001)
    indexDatabase = np.arange(121288)
    indexTrain = np.arange(5001)


    train_set = scio.loadmat('/home/crossai/chuchenglong/code/datasets/coco/COCO_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float32)
    train_x = np.array(train_set['I_tr'], dtype=np.float32)
    train_y = np.array(train_set['T_tr'], dtype=np.float32)

    test_set = scio.loadmat('/home/crossai/chuchenglong/code/datasets/coco/COCO_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float32)
    query_x = np.array(test_set['I_te'], dtype=np.float32)
    query_y = np.array(test_set['T_te'], dtype=np.float32)

    db_set = h5py.File('/home/crossai/chuchenglong/code/datasets/coco/COCO_database.mat')
    retrieval_L = np.array(db_set['L_db'], dtype=np.float32).T
    retrieval_x = np.array(db_set['I_db'], dtype=np.float32).T
    retrieval_y = np.array(db_set['T_db'], dtype=np.float32).T


    coco_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    coco_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_feat_len = query_y.shape[1]
    print(txt_feat_len)

    class MSCOCO(torch.utils.data.Dataset):

        def __init__(self, transform=None, target_transform=None, train=True, database=False):
            self.transform = transform
            self.target_transform = target_transform
            if train:
                self.train_labels = train_L 
                self.train_index = indexTrain
                self.txt = train_y
                self.img = train_x

            elif database:

                self.train_labels = retrieval_L
                self.train_index = indexDatabase
                self.txt = retrieval_y
                self.img = retrieval_x
            else:
                self.train_labels = query_L
                self.train_index = indexTest
                self.txt = query_y
                self.img = query_x
    
        def __getitem__(self, index):

            # mscoco = h5py.File(settings.IMG_DIR,'r', libver='latest', swmr=True )
            # img, target = mscoco['IAll'][self.train_index[index]], self.train_labels[index]
            # img = Image.fromarray(np.array(img, dtype=np.uint8))
            # mscoco.close()

            txt = self.txt[index]
            img = self.img[index]
            target = self.train_index[index]
            if self.transform is not None:
                # img = self.transform(img)
                pass
            if self.target_transform is not None:
                # target = self.target_transform(target)
                pass
            return img, txt, target, index
    
        def __len__(self):
            return len(self.train_labels)  