import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD, lr_scheduler
import matplotlib.pyplot as plt

from dataset import PIEDataset, FRDataset, load_FRDataset, preprocess
from models import FRModel
from loss import CELoss, FocalLoss
from utils import parse_args, get_root_logger, evaluate, calc_feat, evaluate_1to1, evaluate_1toN


if __name__ == '__main__':
    if not os.path.exists('out'):
        os.mkdir('out')

    args = parse_args()
    logger = get_root_logger(args)
    logger.info('===> Current Config')
    for k, v in args._get_kwargs():
        logger.info(f'{k} = {v}')

    logger.info('===> Loading Dataset')
    # load dataset
    if args.dataset == 'PIE':
        in_channel, img_size = 1, 64
        train_set, test_set = PIEDataset(), PIEDataset(test=True)
        n_class = train_set.tot_class
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    else:   # elif args.dataset.startswith('FR'):
        in_channel, img_size = 3, 64
        data_path = dict(
            FR94=[f'data/Face Recognition Data/faces94/{sub_class}' for sub_class in ['female', 'male', 'malestaff']],
            FR95=['data/Face Recognition Data/faces95'],
            FR96=['data/Face Recognition Data/faces96'],
            FR_gri=['data/Face Recognition Data/grimace'],
            FR_all=None,
        )
        train_samples, test_samples = load_FRDataset(data_path[args.dataset])
        train_samples = preprocess(train_samples, align=args.align)
        test_samples = preprocess(test_samples, align=args.align)
        train_set, test_set = FRDataset(train_samples), FRDataset(test_samples)
        n_class = train_set.tot_class
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    logger.info(f'size of training set: {train_set.__len__()}')
    logger.info(f'size of testing set: {test_set.__len__()}')

    logger.info('===> Building Model')
    model = FRModel(in_channel, n_class, img_size, pretrained=args.pretrain, backbone=args.model)

    if args.cuda:
        logger.info('===> Setting CUDA')
        model = model.cuda()

    logger.info('===> Setting Optimizer & Loss')
    if args.optim == 'Adam':
        optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8)
    elif args.optim == 'AdamW':
        optim = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8)
    else:
        optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.05)

    sched = lr_scheduler.StepLR(optimizer=optim, step_size=args.sched_step, gamma=.95)

    if args.loss == 'Focal':
        loss_module = FocalLoss()
    else:
        loss_module = CELoss()

    train_loss_epoch = []
    # train_m1_epoch = []
    # train_m2_epoch = []
    # train_m3_epoch = []
    test_m1_epoch = []
    test_m2_epoch = []
    test_m3_epoch = []
    if args.dataset == 'PIE':
        m1, m2, m3 = 'acc', 'rec', 'f1'
    else:
        m1, m2, m3 = 'acc', 'acc_top1', 'acc_top5'

    for epoch_no in range(args.epoch):
        logger.info(f'===> Training Epoch[{epoch_no+1}/{args.epoch}]')
        model.train()
        loss_arr = []
        for batch_no, input_batch in enumerate(train_loader):
            x, y = input_batch
            x = x.to(model.device())
            y = y.to(model.device())

            feat, y_hat = model(x)
            loss = loss_module(y_hat, y.squeeze(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch_no % 20 == 0:
                logger.info(f'Batch-{batch_no} loss:{loss.item()}')

            loss_arr.append(loss.item())

        sched.step()
        train_loss_epoch.append(np.mean(loss_arr))

        model.eval()
        if args.dataset == 'PIE':
            # logger.info(f'===> Eval on Train set at Epoch-{epoch_no+1}')
            # acc, rec, f1 = evaluate(model, train_loader)
            # train_m1_epoch.append(acc)
            # train_m2_epoch.append(rec)
            # train_m3_epoch.append(f1)
            # logger.info(f'acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}')
            logger.info(f'===> Eval on Test set at Epoch-{epoch_no+1}')
            acc, rec, f1 = evaluate(model, test_loader)
            test_m1_epoch.append(acc)
            test_m2_epoch.append(rec)
            test_m3_epoch.append(f1)
            logger.info(f'acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}')
        else:
            # logger.info(f'===> Eval on Train set at Epoch-{epoch_no + 1}')
            # acc = evaluate_one2one(model, train_loader)
            # acc_top1, acc_top5 = evaluate_one2n(model, train_loader)
            # train_m1_epoch.append(acc)
            # train_m2_epoch.append(acc_top1)
            # train_m3_epoch.append(acc_top5)
            # logger.info(f'acc: {acc:.4f}, acc_top1: {acc_top1:.4f}, acc_top5: {acc_top5:.4f}')
            logger.info(f'===> Eval on Test set at Epoch-{epoch_no + 1}')
            feat_arr = calc_feat(model, test_loader, test_set.tot_class)
            acc, th = evaluate_1to1(feat_arr)
            acc_top1, acc_top5 = evaluate_1toN(feat_arr, test_set.tot_class)
            test_m1_epoch.append(acc)
            test_m2_epoch.append(acc_top1)
            test_m3_epoch.append(acc_top5)
            logger.info(f'acc: {acc:.4f}, th: {th:.4f}, acc_top1: {acc_top1:.4f}, acc_top5: {acc_top5:.4f}')

    logger.info('===> Summary')
    logger.info(f'train_loss_epoch: {train_loss_epoch}')
    # logger.info(f'train_{m1}_epoch: {train_m1_epoch}')
    # logger.info(f'train_{m2}_epoch: {train_m2_epoch}')
    # logger.info(f'train_{m3}_epoch: {train_m3_epoch}')
    logger.info(f'test_{m1}_epoch: {test_m1_epoch}')
    logger.info(f'test_{m2}_epoch: {test_m2_epoch}')
    logger.info(f'test_{m3}_epoch: {test_m3_epoch}')

    logger.info('===> Final Report')
    logger.info(f'best test {m1}: {np.max(test_m1_epoch):.6f} at epoch {np.argmax(test_m1_epoch)+1}')
    logger.info(f'best test {m2}: {np.max(test_m2_epoch):.6f} at epoch {np.argmax(test_m2_epoch)+1}')
    logger.info(f'best test {m3}: {np.max(test_m3_epoch):.6f} at epoch {np.argmax(test_m3_epoch)+1}')

    # painting
    x_axis = [i + 1 for i in range(args.epoch)]

    plt.figure()
    plt.plot(x_axis, train_loss_epoch, marker='.', color='y', label='loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Train Loss Curve')
    plt.savefig(f'out/{args.name}/train_loss_curve.png')
    plt.show()
    plt.cla()

    # plt.figure()
    # plt.plot(x_axis, train_m1_epoch, marker='o', color='r', label=m1)
    # plt.plot(x_axis, train_m2_epoch, marker='s', color='b', label=m2)
    # plt.plot(x_axis, train_m3_epoch, marker='^', color='g', label=m3)
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('value')
    # plt.title('Train Metric Curve')
    # plt.savefig(f'out/{args.name}/train_metric_curve.png')
    # plt.show()
    # plt.cla()

    plt.figure()
    plt.plot(x_axis, test_m1_epoch, marker='o', color='r', label=m1)
    plt.plot(x_axis, test_m2_epoch, marker='s', color='b', label=m2)
    plt.plot(x_axis, test_m3_epoch, marker='^', color='g', label=m3)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Test Metric Curve')
    plt.savefig(f'out/{args.name}/test_metric_curve.png')
    plt.show()
    plt.cla()
