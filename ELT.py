# Credits: https://github.com/thuml/Transfer-Learning-Library
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
import wandb
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.append('../')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM
from dalib.modules.entropy import entropy
from dalib.adaptation.dann import DomainAdversarialLoss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append('.')
import utils
def mydiscriminablity(source_feature,source_pres):
    # source_feature=TSNE(n_components=2, random_state=33).fit_transform(source_feature)
    features=source_feature
    labels=source_pres
    unique_labels = torch.unique(labels)  # 获取唯一的标签值

    class_means = []  # 存储每个类别的特征平均值
    class_variances = []  # 存储每个类别的特征方差
    class_num=[]
    for label in unique_labels:
        indices = torch.where(labels == label)  # 获取具有当前标签的特征的索引
        class_samples = features[indices]  # 获取对应于当前标签的特征样本

        class_mean = torch.mean(class_samples, axis=0)  # 计算某个类别特征样本的平均值
        class_variance=((class_samples-class_samples.mean(0))**2).mean()
        # class_variance = torch.var(class_samples, axis=0)  # 计算某个类别特征样本的方差

        class_means.append(class_mean.unsqueeze_(0)) 
        class_variances.append(class_variance.unsqueeze_(0))
        class_num.append((labels == label).sum().unsqueeze_(0))         #记录当前类别的数量
    class_means=torch.cat(class_means)
    class_variances=torch.cat(class_variances)
    class_num=torch.cat(class_num)
    weight=class_num[:,None]/class_num.sum()
    between_c=torch.mul((class_means-source_feature.mean(0))**2,weight).mean()
    within_c=torch.mul(class_variances,weight).mean()
    return between_c/within_c

def feature2heatmap(features,grads):
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    features=features*pooled_grads
    heatmap_d = features.cpu().detach().numpy()
    heatmap_d = np.mean(heatmap_d, axis=0)
    # heatmap_d = heatmap_d-heatmap_d.mean()
    heatmap_d = np.maximum(heatmap_d, 0)
    # heatmap_d = abs(heatmap_d)
    heatmap_d /= (np.max(heatmap_d)+0.0000001)
    return heatmap_d

def calculate_class_stats(features, labels):
    unique_labels = np.unique(labels)  # 获取唯一的标签值

    class_means = []  # 存储每个类别的特征平均值
    class_variances = []  # 存储每个类别的特征方差
    class_num=[]

    for label in unique_labels:
        indices = np.where(labels == label)  # 获取具有当前标签的特征的索引
        class_samples = features[indices]  # 获取对应于当前标签的特征样本

        class_mean = np.mean(class_samples, axis=0)  # 计算特征样本的平均值
        class_variance = np.var(class_samples, axis=0)  # 计算特征样本的方差

        class_means.append(class_mean) 
        class_variances.append(class_variance)
        class_num.append((labels == label).sum())
        # class_unique_var+=class_variance.mean()
    return np.array(class_means), np.array(class_variances),np.array(class_num)

def main(args: argparse.Namespace, eps=0.):
    logger = CompleteLogger(args.log, args.phase)

    if args.log_results:
        wandb.init(project="1109", name=args.log_name)
        wandb.config.update(args,allow_val_change=True)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    device = args.device

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    # print("train_transform: ", train_transform)
    # print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source,
                          args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    # print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    # print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    pesude_label_generator=nn.Sequential(nn.Linear(classifier.features_dim, num_classes)).to(device)
    pesude_label_generator_weak=nn.Sequential(nn.Linear(classifier.features_dim, num_classes)).to(device)

    if args.randomized:
        domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
        # domain_discri2 = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
        # domain_discri2 = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
    domain_discri_used_for_attention = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    domain_discri2 = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    base_optimizer = torch.optim.SGD
    ad_optimizer = SGD(domain_discri.get_parameters()+domain_discri_used_for_attention.get_parameters()+domain_discri2.get_parameters()+[{"params": pesude_label_generator.parameters(),"lr": 1.0}]\
                    +[{"params": pesude_label_generator_weak.parameters(),"lr": 1.0}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer = SAM(classifier.get_parameters(), base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_ad = LambdaLR(
        ad_optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim, eps=eps
    ).to(device)
    # domain_adv2 = ConditionalDomainAdversarialLoss(
    #     domain_discri2, entropy_conditioning=args.entropy,
    #     num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
    #     randomized_dim=args.randomized_dim, eps=eps
    # ).to(device)
    domain_adv2 = DomainAdversarialLoss(domain_discri2,sigmoid=True).to(device)

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(
            logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    # classifier.load_state_dict(torch.load("Domain_Adaptation/logs/cdan_mcc_sdat_seperated/OfficeHome_Ar2Pr/checkpoints/best.pth", map_location='cpu'))
    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(
            classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature, source_labels, pres_source = collect_feature(
            train_source_loader, classifier, feature_extractor,device)
        target_feature, target_labels ,pres_target= collect_feature(
            train_target_loader, classifier, feature_extractor,device)
        source_feature.requires_grad=True
        target_feature.requires_grad=True

        # my_hok=source_feature.detach().clone()
        features_grad_class=torch.autograd.grad(pow(torch.max(classifier.head(source_feature),1)[0],2).sum(),source_feature,create_graph=True,retain_graph=True,allow_unused=True)[0]
        features_grad_class[features_grad_class>=0]=1
        features_grad_class[features_grad_class<0]=-1
        f_class = (features_grad_class.detach()*source_feature).cpu()
        heatmap_c_strong_s= torch.maximum(f_class, torch.zeros(f_class.size()))
        heatmap_c_weak_s= -torch.minimum(f_class, torch.zeros(f_class.size()))

        # my_hok=source_feature.detach().clone()
        features_grad_class=torch.autograd.grad(pow(torch.max(classifier.head(target_feature),1)[0],2).sum(),target_feature,create_graph=True,retain_graph=True,allow_unused=True)[0]
        features_grad_class[features_grad_class>=0]=1
        features_grad_class[features_grad_class<0]=-1
        f_class = (features_grad_class.detach()*target_feature).cpu()
        heatmap_c_strong_t= torch.maximum(f_class, torch.zeros(f_class.size()))
        heatmap_c_weak_t= -torch.minimum(f_class, torch.zeros(f_class.size()))

        torch.save(source_feature.cpu().detach(),args.root+"/source_feature.pth")
        print("Saving feature to", args.root+"/source_feature.pth")

        torch.save(target_feature.cpu().detach(),args.root+"/target_feature.pth")
        torch.save(heatmap_c_strong_s.cpu().detach(),args.root+"/heatmap_c_strong_s.pth")
        torch.save(heatmap_c_strong_t.cpu().detach(),args.root+"/heatmap_c_strong_t.pth")
        torch.save(heatmap_c_weak_s.cpu().detach(),args.root+"/heatmap_c_weak_s.pth")
        torch.save(heatmap_c_weak_t.cpu().detach(),args.root+"/heatmap_c_weak_t.pth")

        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature.cpu().detach(), target_feature.cpu().detach(), tSNE_filename)
        tSNE_filename_strong = osp.join(logger.visualize_directory, 'TSNE_strong.pdf')
        tsne.visualize(heatmap_c_strong_s.cpu().detach(), heatmap_c_strong_t.cpu().detach(), tSNE_filename_strong)
        tSNE_filename_weak= osp.join(logger.visualize_directory, 'TSNE_weak.pdf')
        tsne.visualize(heatmap_c_weak_s.cpu().detach(), heatmap_c_weak_t.cpu().detach(), tSNE_filename_weak)
        print("Saving t-SNE to", tSNE_filename)

        # calculate A-distance, which is a measure for distribution discrepancy
        # A_distance = a_distance.calculate(
        #     source_feature, target_feature, device)
        A_distance = a_distance.calculate(
            source_feature.detach(), target_feature.detach(), device)
        print("A-distance =", A_distance,"A-distance_strong =", a_distance.calculate(
            heatmap_c_strong_s.detach(), heatmap_c_strong_t.detach(), device),"A-distance_weak =", a_distance.calculate(
            heatmap_c_weak_s.detach(), heatmap_c_weak_t.detach(), device))

        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return
    
        # start training
    best_acc1 = 0.
    if args.loss_consistent:
        mem_fea_strong = torch.rand(num_classes, args.bottleneck_dim).cuda()
        mem_fea_strong = mem_fea_strong / torch.norm(mem_fea_strong, p=2, dim=1, keepdim=True)
        mem_fea_weak = torch.rand(num_classes, args.bottleneck_dim).cuda()
        mem_fea_weak = mem_fea_weak / torch.norm(mem_fea_weak, p=2, dim=1, keepdim=True)
        mem_fea=torch.concatenate((mem_fea_strong,mem_fea_weak),axis=0)
    else:
        mem_fea=None
    for epoch in range(args.epochs):
        feature_extractor = nn.Sequential(
            classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature, source_labels, pres_source = collect_feature(train_source_loader, classifier, feature_extractor,device,max_num_features=1000)
        target_feature, target_labels ,pres_target= collect_feature(train_target_loader, classifier, feature_extractor,device,max_num_features=1000)
        source_feature.requires_grad=True
        target_feature.requires_grad=True

        # my_hok=source_feature.detach().clone()
        features_grad_class=torch.autograd.grad(pow(torch.max(classifier.head(source_feature),1)[0],2).sum(),source_feature,allow_unused=True)[0].detach()
        features_grad_class[features_grad_class>=0]=1
        features_grad_class[features_grad_class<0]=-1
        f_class = (features_grad_class*source_feature).cpu()
        heatmap_c_strong_s= torch.maximum(f_class, torch.zeros(f_class.size()))
        heatmap_c_weak_s= -torch.minimum(f_class, torch.zeros(f_class.size()))

        # my_hok=source_feature.detach().clone()
        features_grad_class=torch.autograd.grad(pow(torch.max(classifier.head(target_feature),1)[0],2).sum(),target_feature,allow_unused=True)[0].detach()
        features_grad_class[features_grad_class>=0]=1
        features_grad_class[features_grad_class<0]=-1
        f_class = (features_grad_class*target_feature).cpu()
        heatmap_c_strong_t= torch.maximum(f_class, torch.zeros(f_class.size()))
        heatmap_c_weak_t= -torch.minimum(f_class, torch.zeros(f_class.size()))
        
        # means, variances,class_nums = calculate_class_stats(np.array(source_feature.cpu().detach()),  np.array(source_labels.cpu().detach()))
        # source_ratio=((((means-np.array(source_feature.cpu().detach()).mean(0))**2).mean(-1)*class_nums).mean()/variances.mean())
        # print("1",(((means-np.array(source_feature.cpu().detach()).mean(0))**2).mean(-1)*class_nums).mean(),"2",variances.mean(),"source_ratio",source_ratio)
        # means, variances,class_nums = calculate_class_stats(np.array(target_feature.cpu().detach()),  np.array(pres_target.cpu().detach().argmax(-1)))
        # target_ratio=((((means-np.array(target_feature.cpu().detach()).mean(0))**2).mean(-1)*class_nums).mean()/variances.mean())
        # print("1",(((means-np.array(target_feature.cpu().detach()).mean(0))**2).mean(-1)*class_nums).mean(),"2",variances.mean(),"target_ratio",target_ratio)
        # print("target_ratio",target_ratio)
        source_strong=mydiscriminablity(heatmap_c_strong_s.cpu(),pres_source.argmax(-1).cpu()).detach().numpy()
        target_strong=mydiscriminablity(heatmap_c_strong_t.cpu(),pres_target.argmax(-1).cpu()).detach().numpy()
        source_weak=mydiscriminablity(heatmap_c_weak_s.cpu(),pres_source.argmax(-1).cpu()).detach().numpy()
        target_weak=mydiscriminablity(heatmap_c_weak_t.cpu(),pres_target.argmax(-1).cpu()).detach().numpy()
        strong_gap=(source_strong-target_strong)/(source_strong+target_strong)
        weak_gap=(source_weak-target_weak)/(source_weak+target_weak)
        # args.parm_seperation=max(2*min(max(weak_gap,0)/(strong_gap),0.5),0)
        args.parm_seperation_strong=args.parm_seperation*max(strong_gap,0)/(max(strong_gap,0)+max(weak_gap,0)+0.0001)
        args.parm_seperation_weak=args.parm_seperation*max(weak_gap,0)/(max(strong_gap,0)+max(weak_gap,0)+0.0001)
        # args.parm_seperation_strong=np.exp(strong_gap)/(np.exp(strong_gap)+np.exp(weak_gap))
        # args.parm_seperation_weak=np.exp(weak_gap)/(np.exp(strong_gap)+np.exp(weak_gap))
        if(epoch==0):
            args.parm_seperation_strong=0.5*args.parm_seperation
            args.parm_seperation_weak=0.5*args.parm_seperation
        args.parm_seperation_strong=1
        args.parm_seperation_weak=1    
        print("source_strong",source_strong,"target_strong",target_strong,\
        "source_weak",source_weak,"target_weak",target_weak)
        print("weak_gap",weak_gap,"strong_gap",strong_gap,)
        print("args.parm_seperation_strong",args.parm_seperation_strong,
        "args.parm_seperation_weak",args.parm_seperation_weak)
        train(train_source_iter, train_target_iter, classifier, domain_adv, mcc_loss, optimizer, ad_optimizer,
              lr_scheduler, lr_scheduler_ad, epoch, args,domain_discri_used_for_attention,domain_adv2,pesude_label_generator,pesude_label_generator_weak,mem_fea)
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        if args.log_results:
            wandb.log({'epoch': epoch, 'val_acc': acc1,'best_acc': best_acc1})
        print("best_acc1 = {:3.1f}".format(best_acc1))
        

    # # evaluate on test set
    # classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    # acc1 = utils.validate(test_loader, classifier, args, device)
    # print("test_acc1 = {:3.1f}".format(acc1))
    # if args.log_results:
    #     wandb.log({'epoch': epoch, 'test_acc': acc1})
    wandb.finish()
    return best_acc1

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: ConditionalDomainAdversarialLoss, mcc, optimizer, ad_optimizer,
          lr_scheduler: LambdaLR, lr_scheduler_ad, epoch: int, args: argparse.Namespace,domain_discri_used_for_attention,domain_adv2:DomainAdversarialLoss,pesude_label_generator,pesude_label_generator_weak,mem_fea):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    target_cls_accs = AverageMeter('target_cls_accs Acc', ':3.1f')
    target_cls_accs_strong = AverageMeter('target_cls_accs_strong Acc', ':3.1f')
    target_cls_accs_weak = AverageMeter('target_cls_accs_weak Acc', ':3.1f')
    target_cls_accs_strong_weak0 = AverageMeter('target_cls_accs_strong_weak0 Acc', ':3.1f')
    target_cls_accs_strong_weak1 = AverageMeter('target_cls_accs_strong_weak1 Acc', ':3.1f')
    target_cls_accs_strong_weak2 = AverageMeter('target_cls_accs_strong_weak2 Acc', ':3.1f')
    # target_cls_accs_strong_weak = AverageMeter('target_cls_accs_strong_weak Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    domain_accs2 = AverageMeter('Domain Acc2', ':3.1f')

    loss_consistents = AverageMeter('loss_consistent', ':1.4f')
    loss_attentions = AverageMeter('loss_attention', ':1.4f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs,domain_accs2,loss_consistents,loss_attentions,target_cls_accs,target_cls_accs_strong,\
         target_cls_accs_weak,target_cls_accs_strong_weak0,target_cls_accs_strong_weak1,target_cls_accs_strong_weak2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()
    domain_discri_used_for_attention.train()
    domain_adv2.train()
    pesude_label_generator.train()
    pesude_label_generator_weak.train()
    end = time.time()
    KLDivLoss = nn.KLDivLoss(reduction='none')

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t=labels_t.to(device)
        d_label = torch.cat((
            torch.ones((x_s.size(0),1)).to(x_s.device),
            torch.zeros((x_t.size(0),1)).to(x_t.device),
        ))
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        ad_optimizer.zero_grad()

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        #+pesude_cls_loss_S+pesude_cls_loss_T+pesude_cls_loss_S_weak
        loss =0
        loss +=cls_loss+ mcc(y_t)
        loss.backward()

        # Calculate ϵ̂ (w) and add it to the weights
        optimizer.first_step(zero_grad=True)        #记录norm等信息 然后清空梯度

        # Calculate task loss and domain loss
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        features_grad_class=torch.autograd.grad(torch.max(y,1)[0].sum(),f,create_graph=True,retain_graph=True,)[0].detach()
        optimizer.zero_grad()

        features_grad_class[features_grad_class>=0]=1
        features_grad_class[features_grad_class<0]=-1
        f_class = features_grad_class*f
        heatmap_c_strong= torch.maximum(f_class, torch.zeros(f_class.size()).to(device))
        heatmap_c_weak= -torch.minimum(f_class, torch.zeros(f_class.size()).to(device))
        loss=0
        cls_loss = F.cross_entropy(y_s, labels_s)

        if args.loss_consistent:
            #计算不同的loss
            mem_fea_strong,mem_fea_weak\
                                = mem_fea.chunk(2, dim=0)

            mem_fea_strong_norm = mem_fea_strong / (torch.norm(mem_fea_strong, p=2, dim=1, keepdim=True)+ 1e-8)
            dis_strong          = torch.mm(heatmap_c_strong.chunk(2, dim=0)[1], mem_fea_strong_norm.t())
            _, pred_strong      = torch.max(dis_strong, dim=1)
            mem_fea_weak_norm   = mem_fea_weak / (torch.norm(mem_fea_weak, p=2, dim=1, keepdim=True)+ 1e-8)
            dis_weak            = torch.mm(heatmap_c_weak.chunk(2, dim=0)[1], mem_fea_weak_norm.t())
            _, pred_weak        = torch.max(dis_weak, dim=1)
            #**************************************************
            # loss_consistent     = nn.CrossEntropyLoss()(dis_weak,pred_strong)
            # weight_strong       = nn.CrossEntropyLoss(reduction="none")(dis_strong,pred_weak)+3
            # weight_strong       = weight_strong/ torch.sum(weight_strong) * len(weight_strong)

            # weight_weak         = nn.CrossEntropyLoss(reduction="none")(dis_weak,pred_strong)+1
            # weight_weak         = weight_weak/ torch.sum(weight_weak) * len(weight_weak)

            p_output            = F.softmax(dis_strong,dim=-1)
            q_output            = F.softmax(dis_weak,dim=-1)
            log_mean_output     = ((p_output + q_output )/2).log()
            distance            = F.cross_entropy(dis_weak,pred_strong,reduction="none")
            #(KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
            distance_adv        = distance.detach()+10
            distance_adv        = distance_adv/ torch.sum(distance_adv) * len(distance_adv)

            distance_self       = 1/(distance+10)
            distance_self       = distance_self/ torch.sum(distance_self) * len(distance_self)
            distance_self       = distance_self.detach()
            weight_weak         = (pred_weak==pred_strong)+1
            weight_weak         = 1/weight_weak
            # weight_strong       = (pred_weak==pred_strong)+1
            weight_weak         = weight_weak/torch.sum(weight_weak)*len(weight_weak)
            # weight              =weight.cpu().detach().numpy()
            # (nn.Softmax(dim=1)(dis_weak)-nn.Softmax(dim=1)(dis_strong)).abs().mean()*100
            # loss                +=loss_consistent*0.1
            target_acc_strong   = accuracy(dis_strong,labels_t)[0]
            target_acc_weak     = accuracy(dis_weak,labels_t)[0]
            if(dis_strong[pred_strong==pred_weak].size(0)>0):
                target_acc_strong_weak0\
                                = accuracy(dis_strong[pred_strong==pred_weak],labels_t[pred_strong==pred_weak])[0]
                target_cls_accs_strong_weak0.update(target_acc_strong_weak0, dis_strong[pred_strong==pred_weak].size(0))
            target_cls_accs_strong.update(target_acc_strong, x_s.size(0))
            target_cls_accs_weak.update(target_acc_weak, x_s.size(0))

            #更新新的聚类中心
            with torch.no_grad():
                softmax_t       = nn.Softmax(dim=1)(y_t).detach()
                _, pred_t       = torch.max(softmax_t, 1)
                onehot_t        = torch.eye(softmax_t.shape[1]).cuda()[pred_t]
                momentum        = 0.1
                center_t_strong = torch.mm(heatmap_c_strong.chunk(2, dim=0)[1].detach().t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)
                mem_fea_strong  = (1.0 - momentum) * mem_fea_strong + momentum * center_t_strong.t().clone()

                center_t_weak   = torch.mm(heatmap_c_weak.chunk(2, dim=0)[1].detach().t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)
                mem_fea_weak    = (1.0 - momentum) * mem_fea_weak   + momentum * center_t_weak.t()  .clone()
                mem_fea         = torch.concatenate((mem_fea_strong,mem_fea_weak),axis=0).detach()

                # softmax         = nn.Softmax(dim=1)(y).detach()
                # _, pred         = torch.max(softmax, 1)
                # onehot_s        = torch.eye(softmax.shape[1]).cuda()[pred]
                # momentum        = 0.1
                # center_s_strong = torch.mm(heatmap_c_strong.detach().t(), onehot_s) / (onehot_s.sum(dim=0) + 1e-8)
                # mem_fea_strong  = (1.0 - momentum) * mem_fea_strong + momentum * center_s_strong.t().clone()

                # center_s_weak   = torch.mm(heatmap_c_weak.detach().t(), onehot_s) / (onehot_s.sum(dim=0) + 1e-8)
                # mem_fea_weak    = (1.0 - momentum) * mem_fea_weak   + momentum * center_s_weak.t()  .clone()

                # mem_fea         = torch.concatenate((mem_fea_strong,mem_fea_weak),axis=0).detach()

        g = F.softmax(y, dim=1).detach()
        weight = 1.0 + torch.exp(-entropy(g))
        weight=weight / torch.sum(weight) * len(weight)
        source_weight,target_weight=weight.chunk(2, dim=0)
        if args.align_seperation:
            if args.loss_consistent:
                transfer_loss =args.parm_seperation_strong*domain_adv(y_s, heatmap_c_strong.chunk(2, dim=0)[0], y_t, heatmap_c_strong.chunk(2, dim=0)[1],myweight=weight_weak)+\
                args.parm_seperation_weak*domain_adv2(heatmap_c_weak.chunk(2, dim=0)[0], heatmap_c_weak.chunk(2, dim=0)[1],myweight=weight_weak)
                #,w_s=source_weight,w_t=target_weight)
                mcc_value=mcc(y_t)
            else:
                transfer_loss =args.parm_seperation_strong*domain_adv(y_s, heatmap_c_strong.chunk(2, dim=0)[0], y_t, heatmap_c_strong.chunk(2, dim=0)[1])+\
                args.parm_seperation_weak*domain_adv2(heatmap_c_weak.chunk(2, dim=0)[0], heatmap_c_weak.chunk(2, dim=0)[1])
                #,w_s=source_weight,w_t=target_weight)
                mcc_value=mcc(y_t)
            domain_acc2 = domain_adv2.domain_discriminator_accuracy
            domain_accs2.update(domain_acc2, x_s.size(0)) 
        else:
            if args.loss_consistent:
                transfer_loss =domain_adv(y_s, f.chunk(2, dim=0)[0], y_t, f.chunk(2, dim=0)[1])
                mcc_value=mcc(y_t)
            else:
                transfer_loss =domain_adv(y_s, f.chunk(2, dim=0)[0], y_t, f.chunk(2, dim=0)[1])
                mcc_value=mcc(y_t)

        loss += cls_loss + transfer_loss * args.trade_off+mcc_value
        #+pesude_cls_loss_S+pesude_cls_loss_T+pesude_cls_loss_S_weak
        domain_acc = domain_adv.domain_discriminator_accuracy

        cls_acc = accuracy(y_s, labels_s)[0]
        target_acc=accuracy(y_t,labels_t)[0]

        if args.log_results:
            wandb.log({'iteration': epoch*args.iters_per_epoch + i, 'loss': loss, 'cls_loss': cls_loss,
                       'transfer_loss': transfer_loss, 'domain_acc1': domain_adv.domain_discriminator_accuracy,'domain_acc2': domain_adv2.domain_discriminator_accuracy})
                       #,"loss_consistent":loss_consistent})
                       #,"loss_attention":loss_attention})

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        domain_accs.update(domain_acc, x_s.size(0))
        target_cls_accs.update(target_acc, x_s.size(0))
        # domain_accs2.update(domain_acc2, x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        # loss_consistents.update(loss_consistent.item(), x_s.size(0))
        # loss_attentions.update(loss_attention.item(), x_s.size(0))
        loss.backward()
        # Update parameters of domain classifier
        ad_optimizer.step()
        # Update parameters (Sharpness-Aware update)
        optimizer.second_step(zero_grad=True)
        lr_scheduler.step()
        lr_scheduler_ad.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            img = x_s[0][None,:]
            model.eval()
            domain_discri_used_for_attention.eval()
            features = model.features(img)
            y_d= domain_discri_used_for_attention(model.domain_feature(features))
            f, y = model.classifier(features)
            features_grad_class=torch.autograd.grad(torch.max(y,1)[0].sum(),features,create_graph=True,retain_graph=True,)[0]
            features_grad_domain=torch.autograd.grad(torch.max(y_d,1)[0].sum(),features,create_graph=True,retain_graph=True,)[0]
            features_grad_domain1=torch.autograd.grad(torch.min(y_d,1)[0].sum(),features,create_graph=True,retain_graph=True,)[0]
            heatmap_c= feature2heatmap(features,features_grad_class)
            heatmap_d= feature2heatmap(features,features_grad_domain)
            heatmap_d1= feature2heatmap(features,features_grad_domain1)

            heatmap_c = cv2.resize(heatmap_c, (224, 224))
            heatmap_c = cv2.applyColorMap( np.uint8(heatmap_c*255), cv2.COLORMAP_JET)
            heatmap_d = cv2.resize(heatmap_d, (224, 224))
            heatmap_d = cv2.applyColorMap( np.uint8(heatmap_d*255), cv2.COLORMAP_JET)
            heatmap_d1 = cv2.resize(heatmap_d1, (224, 224))
            heatmap_d1 = cv2.applyColorMap( np.uint8(heatmap_d1*255), cv2.COLORMAP_JET)
            superimposed_img_c = heatmap_c/255*3 + img.cpu().numpy()[0].transpose(1,2,0)
            superimposed_img_d = heatmap_d/255*3  + img.cpu().numpy()[0].transpose(1,2,0)
            superimposed_img_d1 = heatmap_d1/255*3  + img.cpu().numpy()[0].transpose(1,2,0)
            Img= wandb.Image(img, caption=args.class_names[labels_s[0]])
            Img_c= wandb.Image(superimposed_img_c , caption=args.class_names[labels_s[0]])
            Img_d= wandb.Image(superimposed_img_d , caption=args.class_names[labels_s[0]])
            Img_d1= wandb.Image(superimposed_img_d1 , caption=args.class_names[labels_s[0]])

            wandb.log({"Attention_C": Img_c,"Attention_D_max": Img_d,"Attention_D_min": Img_d1,})
            
            # img = x_s[0][None,:]
            # model.eval()
            # features = model.features(img)
            # f, y = model.classifier(features)
            # features_grad_class=torch.autograd.grad(torch.max(y,1)[0].sum(),f,create_graph=True,retain_graph=True,)[0].detach()
            features_grad_class=torch.autograd.grad(torch.max(y,1)[0].sum(),f,create_graph=True,retain_graph=True)[0].detach()
            
            heatmap_c_strong=torch.ones_like(features_grad_class)*(features_grad_class>0)*f
            heatmap_c_weak=torch.ones_like(features_grad_class)*(features_grad_class<0)*f
            PTCF=torch.autograd.grad(heatmap_c_strong.sum(),features,create_graph=True,retain_graph=True)[0].detach()
            PTDF=torch.autograd.grad(heatmap_c_weak.sum(),features,create_graph=True,retain_graph=True)[0].detach()

            heatmap_PTCF= feature2heatmap(features,PTCF)
            heatmap_PTDF= feature2heatmap(features,PTDF)
            heatmap_c = cv2.resize(heatmap_PTCF, (224, 224))
            heatmap_c = cv2.applyColorMap( np.uint8(heatmap_c*255), cv2.COLORMAP_JET)
            heatmap_d = cv2.resize(heatmap_PTDF, (224, 224))
            heatmap_d = cv2.applyColorMap( np.uint8(heatmap_d*255), cv2.COLORMAP_JET)

            superimposed_img_c = heatmap_c/255*3 + img.cpu().numpy()[0].transpose(1,2,0)
            superimposed_img_d = heatmap_d/255*3  + img.cpu().numpy()[0].transpose(1,2,0)
            Img= wandb.Image(img, caption=args.class_names[labels_s[0]])
            Img_c= wandb.Image(superimposed_img_c , caption=args.class_names[labels_s[0]])
            Img_d= wandb.Image(superimposed_img_d , caption=args.class_names[labels_s[0]])

            wandb.log({"Raw_image":Img,"heatmap_PTCF": Img_c,"heatmap_PTDF": Img_d,})
            progress.display(i)
            model.train()
            domain_discri_used_for_attention.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('-root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--eps', default=0.0, type=float,
                        help='hyper-parameter for environemnt label smoothing.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False,
                        action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--log_results', action='store_true',
                        help="To log results in wandb")
    parser.add_argument('--gpu', type=str, default="0", help="GPU ID")
    parser.add_argument('--log_name', type=str,
                        default="log", help="log name for wandb")
    parser.add_argument('--rho', type=float, default=0.02, help="GPU ID")
    parser.add_argument('--temperature', default=2.0,
                        type=float, help='parameter temperature scaling')
    parser.add_argument('--align_seperation',action='store_true',
                        help='using align_seperation')    
    parser.add_argument('--parm_seperation', default=1,
                        type=float, help='momentum')
    parser.add_argument('--parm_seperation_strong', default=0.2,
                        type=float,) 
    parser.add_argument('--parm_seperation_weak', default=0.2,
                        type=float,)    
    parser.add_argument('--loss_consistent',action='store_true',
                        help='using loss_consistent')
    parser.add_argument('--parm_consistent', default=1,
                        type=float, help='momentum')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    import numpy as np
    # epses = [0,0.2,0.4,0.6,0.8,1]
    main(args)

    # # epses = [1]
    # acces = np.array([0. for i in epses])
    # for i in range(len(epses)):
    #     print('eps as this epoch is', epses[i])
    #     args.parm_seperation=epses[i]
    #     acces[i] = main(args)
    # idx = acces.argmax()
    # print(f'the best result is {acces[idx]} with eps {epses[idx]}')
    # print(acces)
