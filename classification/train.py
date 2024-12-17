import argparse
import os
from datetime import datetime

from dataset import get_dataloader
from utils import *
from tqdm import tqdm


def train_one_epoch(train_loader, model, criterion, optimizer, cur_epoch, cutmix_or_mixup):
    # switch to train mode
    model.train()

    pbar = tqdm(train_loader, ncols=150)
    pbar.set_description(f"Training @ Epoch: {cur_epoch}")

    for data in pbar:
        # get the inputs; data is a list of [inputs, targets]
        inputs, targets = data
        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs, targets = cutmix_or_mixup(inputs, targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        pbar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']: .5f}",
            loss=f"{loss.item(): .5f}"
        )

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

    # torch.save(model.state_dict(), f"./checkpoints/_epoch_{epoch + 1}.pth")


@torch.no_grad()
def test(test_dataloader, model, criterion, cur_epoch):
    N = len(test_dataloader.dataset)
    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    pbar = tqdm(test_dataloader, ncols=150)
    pbar.set_description(f"Testing @ Epoch: {cur_epoch}")
    for data in pbar:
        inputs, targets = data
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum()

        pbar.set_postfix(
            loss=f"{test_loss: .5f}"
        )
    return correct.item() / N, test_loss


def main(args):
    try:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = f"{args.work_path}/{args.model}_{str(date)}"
        os.makedirs(path)
        print(f"Current work path: {path}")
    except OSError:
        print("Cannot create a work path!")
        return
    # Store model
    state = {}
    # Get dataloader
    train_dataloader, test_dataloader = get_dataloader(args)
    # Init model
    model = get_model(args).cuda()
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warm, eta_min=args.lr_min
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warm
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warm]
    )
    # data augmentation
    from utils import num_classes_dict
    num_classes = num_classes_dict[args.dataset]
    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    results = []
    # Init values
    best_acc = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train_one_epoch(train_dataloader, model, criterion, optimizer, epoch + 1, cutmix_or_mixup)
        # step
        lr_scheduler.step()
        # evaluate on test set
        acc, avg_loss = test(test_dataloader, model, criterion, epoch + 1)

        results.append((acc, avg_loss))
        # remember best prec@1 and save checkpoint if desired
        if acc > best_acc:
            best_acc = acc
            state = {
                'model': model.state_dict(),
                'best_acc': acc,
                'best_epoch': epoch + 1
            }
        # print current epoch's result
        print(f"=========== Epoch {epoch + 1} Summary ===========")
        print(f"Epoch Acc: {acc}")
        print(f"Best Acc: {best_acc}")
        print(f"=======================================")
    # save result file
    csv_file_name = save_results_csv(path, args.model, results)
    print(f"Result file is saved in {csv_file_name}!")
    # save model file
    pth_file_name = save_model(path, args.model, state)
    print(f"Best model is saved in {pth_file_name}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='model type')
    parser.add_argument('-dataset', type=str, default='cifar100', help='dataset type')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-epochs', type=int, default=300, help='epochs for training')
    parser.add_argument('-warm', type=int, default=20, help='epochs for warm')
    parser.add_argument('-lr_min', type=float, default=1e-6, help='minimal learning rate')
    parser.add_argument('-num_workers', type=int, default=4, help='the number of worker for dataloader')
    parser.add_argument('-work_path', type=str, default='./results', help='work path for files storage')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    main(args)
