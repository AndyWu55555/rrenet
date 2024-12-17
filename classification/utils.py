from rrenet import *


def get_parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_classes_dict = {
    'cifar10': 10,
    'cifar100': 100
}


def get_model(args):
    num_classes = num_classes_dict[args.dataset]
    if args.model == 'c2_rre_n':
        m = get_rrenet(size='n', g_order=2, rre=True, num_classes=num_classes)
    elif args.model == 'c2_sre_n':
        m = get_rrenet(size='n', g_order=2, rre=False, num_classes=num_classes)
    elif args.model == 'c4_rre_n':
        m = get_rrenet(size='n', g_order=4, rre=True, num_classes=num_classes)
    elif args.model == 'c4_rre_s':
        m = get_rrenet(size='s', g_order=4, rre=True, num_classes=num_classes)
    elif args.model == 'c4_rre_m':
        m = get_rrenet(size='m', g_order=4, rre=True, num_classes=num_classes)
    elif args.model == 'c4_sre_n':
        m = get_rrenet(size='n', g_order=4, rre=False, num_classes=num_classes)
    elif args.model == 'c6_rre_n':
        m = get_rrenet(size='n', g_order=6, rre=True, num_classes=num_classes)
    elif args.model == 'c6_sre_n':
        m = get_rrenet(size='n', g_order=6, rre=False, num_classes=num_classes)
    elif args.model == 'c8_rre_n':
        m = get_rrenet(size='n', g_order=8, rre=True, num_classes=num_classes)
    elif args.model == 'c8_sre_n':
        m = get_rrenet(size='n', g_order=8, rre=False, num_classes=num_classes)
    else:
        raise f'Cannot find the model {args.model}'
    print('Calculate model parameters...')
    total_parameters = get_parameter_number(m)
    print(f'Model: {args.model}\nTotal Parameters: {total_parameters}')
    return m


def save_results_csv(path, model_name, results):
    strs = "Epoch, Top-1 Acc, Loss\n"
    for i, result in enumerate(results):
        strs += f"{i + 1}, {result[0]}, {result[1]}\n"
    file_name = f'{path}/{model_name}_result.csv'
    with open(file_name, 'w') as f:
        f.write(strs)
    f.close()
    return file_name


def save_model(path, model_name, state):
    file_name = f'{path}/best_{model_name}_epoch: {str(state["best_epoch"])}_acc: {str(state["best_acc"])}.pth'
    torch.save(state, file_name)
    return file_name
