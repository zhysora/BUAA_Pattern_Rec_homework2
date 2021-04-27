import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from dataset import PIEDataset, FRDataset, load_FRDataset
from models import SwinFR
from utils import parse_args, get_root_logger, evaluate


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
        in_channel = 1
        n_class = 68
        img_size = 64
        train_loader = DataLoader(
            dataset=PIEDataset(),
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=PIEDataset(test=True),
            batch_size=args.batch_size,
            shuffle=False
        )
    elif args.dataset == 'FR':
        train_x, train_y, test_x, test_y, n_class = load_FRDataset()
        in_channel = 3
        img_size = 200
        train_loader = DataLoader(
            dataset=FRDataset(train_x, train_y),
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=FRDataset(test_x, test_y),
            batch_size=args.batch_size,
            shuffle=False
        )

    logger.info('===> Building Model')
    model = SwinFR(in_channel, n_class, img_size)

    if args.cuda:
        logger.info('===> Setting CUDA')
        model = model.cuda()

    logger.info('===> Setting Optimizer & Loss')
    optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8)
    # optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.05)
    sched = lr_scheduler.StepLR(optimizer=optim, step_size=5, gamma=0.95)
    CEloss = CrossEntropyLoss()

    train_loss_epoch = []
    train_acc_epoch = []
    train_rec_epoch = []
    train_f1_epoch = []
    test_acc_epoch = []
    test_rec_epoch = []
    test_f1_epoch = []

    for epoch_no in range(args.epoch):
        logger.info(f'===> Training Epoch[{epoch_no+1}/{args.epoch}]')
        model.train()
        loss_arr = []
        for batch_no, input_batch in enumerate(train_loader):
            x, y = input_batch
            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            y_hat = model(x)
            loss = CEloss(y_hat, y.squeeze(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch_no % 20 == 0:
                logger.info(f'Batch-{batch_no} loss:{loss.item()}')

            loss_arr.append(loss.item())

        sched.step()
        train_loss_epoch.append(np.mean(loss_arr))
        logger.info(f'===> Eval on Train set at Epoch-{epoch_no+1}')
        acc, rec, f1 = evaluate(args, model, train_loader)
        train_acc_epoch.append(acc)
        train_rec_epoch.append(rec)
        train_f1_epoch.append(f1)
        logger.info(f'acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}')
        logger.info(f'===> Eval on Test set at Epoch-{epoch_no+1}')
        acc, rec, f1 = evaluate(args, model, test_loader)
        test_acc_epoch.append(acc)
        test_rec_epoch.append(rec)
        test_f1_epoch.append(f1)
        logger.info(f'acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}')

    logger.info('===> Final Report')
    logger.info(f'best test acc: {np.max(test_acc_epoch):.6f} at epoch {np.argmax(test_acc_epoch)+1}')
    logger.info(f'best test recall: {np.max(test_rec_epoch):.6f} at epoch {np.argmax(test_rec_epoch)+1}')
    logger.info(f'best test f1: {np.max(test_f1_epoch):.6f} at epoch {np.argmax(test_f1_epoch)+1}')

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

    plt.figure()
    plt.plot(x_axis, train_acc_epoch, marker='o', color='r', label='acc')
    plt.plot(x_axis, train_rec_epoch, marker='s', color='b', label='recall')
    plt.plot(x_axis, train_f1_epoch, marker='^', color='g', label='f1')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Train Metric Curve')
    plt.savefig(f'out/{args.name}/train_metric_curve.png')
    plt.show()
    plt.cla()

    plt.figure()
    plt.plot(x_axis, test_acc_epoch, marker='o', color='r', label='acc')
    plt.plot(x_axis, test_rec_epoch, marker='s', color='b', label='recall')
    plt.plot(x_axis, test_f1_epoch, marker='^', color='g', label='f1')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Test Metric Curve')
    plt.savefig(f'out/{args.name}/test_metric_curve.png')
    plt.show()
    plt.cla()
