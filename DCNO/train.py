import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models import *
from utils import *
from T_lossfunc import *
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
from datetime import date, datetime
import scipy.io as scio
import matplotlib.pyplot as plt
from timeit import default_timer
from torchinfo import summary

def train_data(model, train_loader, loss_func, optimizer, lr_scheduler, train_len, losstype, device):
    model.train()
    loss_l2_epoch = 0.
    loss_h1_epoch = 0.

    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        lossl2, lossh1 = loss_func(pred, y)
        if losstype == 'H1':
            lossh1.backward()
        elif losstype == 'L2':
            lossl2.backward()
        optimizer.step()

        loss_l2_epoch += lossl2
        loss_h1_epoch += lossh1

    loss_l2 = loss_l2_epoch / train_len
    loss_h1 = loss_h1_epoch / train_len
    lr = optimizer.param_groups[0]['lr']
    lr_scheduler.step()

    return loss_l2.item(), loss_h1.item(), lr

@torch.no_grad()
def val_data(model, test_loader, loss_func, test_len, device):
    model.eval()
    loss_l2_epoch = 0.
    loss_h1_epoch = 0.

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            lossl2, lossh1 = loss_func(pred, y)
            loss_l2_epoch += lossl2
            loss_h1_epoch += lossh1

    loss_l2 = loss_l2_epoch / test_len
    loss_h1 = loss_h1_epoch / test_len
    return loss_l2.item(), loss_h1.item()

@torch.no_grad()
def test_data(model, test_loader,  res, loss_func, test_len, device):
    pred_rec_step = torch.zeros((1, res, res), device=device)
    model.eval()
    loss_l2_epoch = 0.
    loss_h1_epoch = 0.

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).reshape(-1, res, res)
            pred_rec_step = torch.cat((pred_rec_step, pred), dim=0)
            lossl2, lossh1 = loss_func(pred, y)
            loss_l2_epoch += lossl2
            loss_h1_epoch += lossh1

    loss_l2 = loss_l2_epoch / test_len
    loss_h1 = loss_h1_epoch / test_len

    return pred_rec_step, loss_l2.item(), loss_h1.item()

def train_2D_model(R_dic):

    R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    R_dic['res_output'] = int(
        (R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_stride'] + 1)

    if 'model_name' not in R_dic:
        R_dic['model_name'] = str(R_dic['model']) + 'res' + str(R_dic['res_output']) + '_' + str(R_dic['train_path'])[0:-10] + \
                             '_inv:' + str(R_dic['inv_problem']) + \
                            '_noise' + str(R_dic['noise']) + '_' + str(date.today()) + '.pt'

    R_dic['result_name'] = str(R_dic['model_name'][0:-3]) + '.pkl'
    R_dic['mat_name'] = str(R_dic['model_name'][0:-3]) + '.mat'
    R_dic['fig_name'] = str(R_dic['model_name'][0:-3]) + '.png'

    print('=' * 80)
    print(R_dic)
    print('=' * 80)

    print('=' * 80)
    print('Model:' + str(R_dic['model']))
    print('Train set:' + str(R_dic['train_path']))
    print('Test set:' + str(R_dic['test_path']))
    print('Model save path:' + str(R_dic['model_save_path']))
    print('Train set number:' + str(R_dic['train_len']) + '  ' + 'Test set number:' + str(R_dic['test_len']) + '  '
          + 'Epoch:' + str(R_dic['epochs']))
    print('Inverse problem:' + str(R_dic['inv_problem']) + '  ' + 'Noise:' + str(R_dic['noise'])
          + '  ' + 'Losstype:' + str(R_dic['losstype']))
    print('model name(.pt):' + str(R_dic['model_name']))
    print('=' * 80)

    # ==================== Load data ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join(R_dic['Data_path'], R_dic['train_path'])
    val_path = os.path.join(R_dic['Data_path'], R_dic['val_path'])
    test_path = os.path.join(R_dic['Data_path'], R_dic['test_path'])

    x_train, y_train, x_normalizer, y_normalizer = \
        Data_load(train_path, R_dic['train_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], inv_problem=R_dic['inv_problem'], noise=R_dic['noise'], train_data=True)
    x_val, y_val, _, _ = \
        Data_load(val_path, R_dic['val_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], xnormalizer=x_normalizer, inv_problem=R_dic['inv_problem'],
                  noise=R_dic['noise'], train_data=False)
    x_test, y_test, _, _ = \
        Data_load(test_path, R_dic['test_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], xnormalizer=x_normalizer, inv_problem=R_dic['inv_problem'],
                  noise=R_dic['noise'], train_data=False)

    R_dic['y_norm'] = y_normalizer

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train.contiguous(), y_train.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val.contiguous(), y_val.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test.contiguous(), y_test.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)

    # ==================== set the seed ====================
    seed = R_dic['seed']
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # ==================== Load model, optimizer, learning rate scheduler, loss function ====================
    if 'in_dim' not in R_dic:
        R_dic['in_dim'] = 1
    if R_dic['model'] == 'DCNO2d':
        model = DCNO2d(R_dic)

    summary(model, input_size=(R_dic['batch_size'], R_dic['in_dim'], R_dic['res_input'] , R_dic['res_input'] ))
    print('Parameters number:' + str(count_params(model)))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    # optimizer = Adam(model.parameters(), weight_decay=1e-4)
    lr_scheduler = OneCycleLR(optimizer, max_lr=R_dic['max_lr'],
                              div_factor=R_dic['div_factor'],
                              final_div_factor=R_dic['final_div_factor'],
                              pct_start=R_dic['pct_start'],
                              steps_per_epoch=1,
                              epochs=R_dic['epochs'])

    loss_func = FH1Loss(res=R_dic['res_output'], modes=R_dic['H1loss_modes'])
    loss_func.cuda(device)
    epochs = R_dic['epochs']

    pred_rec_test = torch.zeros(1, R_dic['test_len'], R_dic['res_output'], R_dic['res_output'])
    loss_train = []
    loss_val = []
    loss_test = []
    lr_history = []
    stop_counter = 0
    pred_rec_step_best = None
    best_val_metric = np.inf
    test_by_val = np.inf
    best_val_epoch = None

    Figure_PATH = os.path.join('figures', R_dic['fig_name'])
    lr_name = str('lr_') + R_dic['fig_name']
    lr_PATH = os.path.join('figures', lr_name)

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            train_l2, train_h1, lr = train_data(model, train_loader, loss_func, optimizer, lr_scheduler,
                                                train_len=R_dic['train_len'], losstype=R_dic['losstype'], device=device)
            loss_train.append([train_l2, train_h1])
            lr_history.append(lr)
            val_l2, val_h1 = val_data(model, val_loader, loss_func, test_len=R_dic['val_len'], device=device)
            loss_val.append([val_l2, val_h1])
            pred_rec_step, test_l2, test_h1 = test_data(model, test_loader, R_dic['res_output'], loss_func, test_len=R_dic['test_len'],  device=device)
            loss_test.append([test_l2, test_h1])

            if val_l2 < best_val_metric:
                best_val_epoch = epoch
                best_val_metric = val_l2
                test_by_val = test_l2
                pred_rec_step_best = pred_rec_step
                stop_counter = 0
                torch.save(model, os.path.join(R_dic['model_save_path'], R_dic['model_name']))
            else:
                stop_counter += 1

            if epoch == 0 or (epoch + 1) % 10 == 0:
                pred_rec_step_best = pred_rec_step[1:,:,:].reshape(1, R_dic['test_len'], R_dic['res_output'], R_dic['res_output']).detach().cpu()
                pred_rec_test = torch.cat((pred_rec_test, pred_rec_step_best), dim=0)

            desc = color(f"| Test L2 loss: {test_l2:.3e} ", color=Colors.blue)
            desc += color(f"| test by val: {test_by_val:.3e} at epoch {best_val_epoch + 1}", color=Colors.green)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.green)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)
            desc_ep = color("", color=Colors.red)
            desc_ep += color(f"| Train L2 loss : {train_l2:.3e} ", color=Colors.red)
            desc_ep += color(f"| Val L2 loss : {val_l2:.3e} ", color=Colors.yellow)
            desc_ep += desc
            pbar.set_description(desc_ep)
            pbar.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                test_by_val=test_by_val,
                loss_train=np.asarray(loss_train),
                loss_test=np.asarray(loss_test),
                lr_history=np.asarray(lr_history),
                optimizer_state=optimizer.state_dict()
            )

            save_pickle(result, os.path.join(R_dic['model_save_path'], R_dic['result_name']))
            loss_train_p = np.asarray(loss_train)
            loss_test_p = np.asarray(loss_test)
            x = np.arange(0, epoch + 1, 1)
            plt.style.use('seaborn')
            fig1 = plt.figure()
            plt.semilogy(x, loss_train_p[..., 0], color='red', linestyle="-", linewidth=1, label="train_l2loss")
            plt.semilogy(x, loss_train_p[..., 1], color='red', linestyle="-.", linewidth=1, label="train_h1loss")
            plt.semilogy(x, loss_test_p[..., 0], color='blue', linestyle="-", linewidth=1, label="test_l2loss")
            plt.semilogy(x, loss_test_p[..., 1], color='blue', linestyle="-.", linewidth=1, label="test_h1loss")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(fontsize=15)
            plt.savefig(Figure_PATH)
            plt.close(fig1)
            fig2 = plt.figure()
            lr_history_p = np.asarray(lr_history)
            plt.plot(x, lr_history_p, color='green', linestyle="-", linewidth=1, label="learning rate")
            plt.xlabel("Epochs")
            plt.ylabel("lr")
            plt.legend(fontsize=15)
            plt.savefig(lr_PATH)
            plt.close(fig2)

    print('END')
    print('Test error:' + str(test_by_val))

    s1 = R_dic['res_output']
    s2 = R_dic['res_input']
    z_true, z = torch.zeros(1, s1, s1).to(device), torch.zeros(1, s1, s1).to(device)
    xinput = torch.zeros(1, s2, s2).to(device)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(-1, s1, s1)
            y = y.reshape(-1, s1, s1)
            if R_dic['xGN']:
                x_normalizer = x_normalizer.to(device)
                x = x_normalizer.decode(x.reshape(-1, s2, s2))
            else:
                x = x.reshape(-1, s2, s2)
            xinput = torch.cat((xinput, x), dim=0)
            z_true = torch.cat((z_true, y), dim=0)
            z = torch.cat((z, out), dim=0)
    xinput = xinput[1:, ...].cpu().numpy()
    z_true = z_true[1:, ...].cpu().numpy()
    z = z[1:, ...].cpu().numpy()
    SAVE_PATH = os.path.join('results', R_dic['mat_name'])
    pred_test = pred_rec_test[1:,:,:,:].detach().cpu().numpy()

    scio.savemat(SAVE_PATH, {'input': xinput, 'truth': z_true, 'output': z, 'pred_test':pred_test})# for resolution=512, it may cause error, just comment out this line

    print('Everything is OK!')
    print('Results save path' + str(SAVE_PATH))
    print('Figures save path' + str(Figure_PATH))
    # return result
    return test_by_val


def train_ns(model, train_loader, loss_func, optimizer, lr_scheduler, train_len, T_in, T, device):
    model.train()
    loss_l2_add = 0.
    loss_l2_full = 0.
    for xx, yy in train_loader:
        loss = 0
        batch_size = xx.shape[0]
        optimizer.zero_grad()
        xx = xx.to(device)
        yy = yy.to(device)
        if T>30:
            for t in range(T):
                # x = xx[:, t:t + T_in, :, :]
                x = xx
                y = yy[..., t:t + 1, :, :]
                im = model(x)
                im = im.unsqueeze(1)
                loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)
                xx = torch.cat((xx[:, 1:, :, :], im), dim=1)
        else:
            for t in range(T):
                x = xx[:, t:t + T_in, :, :]
                # x = xx
                y = yy[..., t:t + 1, :, :]
                im = model(x)
                im = im.unsqueeze(1)
                loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)
                # xx = torch.cat((xx[:, 1:, :, :], im), dim=1)
        loss.backward()
        lossl2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

        optimizer.step()
        loss_l2_add += loss
        loss_l2_full += lossl2_full

    loss_l2_add = loss_l2_add / train_len
    loss_l2_full = loss_l2_full / train_len

    lr = optimizer.param_groups[0]['lr']
    lr_scheduler.step()

    return loss_l2_add.item(), loss_l2_full.item(), lr

@torch.no_grad()
def test_ns(model, test_loader, loss_func, test_len, T, device):
    model.eval()
    loss_l2_add = 0.
    loss_l2_full = 0.
    for xx, yy in test_loader:
        loss = 0
        batch_size = xx.shape[0]
        xx = xx.to(device)
        yy = yy.to(device)
        for t in range(T):
            y = yy[..., t:t + 1, :, :]
            im = model(xx)
            im = im.unsqueeze(1)
            loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)

            xx = torch.cat((xx[:, 1:, :, :], im), dim=1)

        lossl2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

        loss_l2_add += loss
        loss_l2_full += lossl2_full

    loss_l2_add = loss_l2_add / test_len
    loss_l2_full = loss_l2_full / test_len

    return loss_l2_add.item(), loss_l2_full.item()

def train_NS_model(R_dic):
    R_dic['in_dim'] = R_dic['T_in']
    R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    R_dic['res_output'] = int(
        (R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_stride'] + 1)
    if 'model_name' not in R_dic:
        R_dic['model_name'] = str(R_dic['model']) + 'res' + str(R_dic['res_output']) + '_' + str(R_dic['train_path'])[
                                                                                             0:-10] + \
                              '_epoch' + str(R_dic['epochs']) + '_' + str(date.today()) + '.pt'
    R_dic['result_name'] = str(R_dic['model_name'][0:-3]) + '.pkl'
    R_dic['mat_name'] = str(R_dic['model_name'][0:-3]) + '.mat'
    R_dic['fig_name'] = str(R_dic['model_name'][0:-3]) + '.png'

    print('=' * 80)
    print(R_dic)
    print('=' * 80)

    print('=' * 80)
    print('Model:' + str(R_dic['model']))
    print('Train set:' + str(R_dic['train_path']))
    print('Test set:' + str(R_dic['test_path']))
    print('Model save path:' + str(R_dic['model_save_path']))
    print('Train set number:' + str(R_dic['train_len']) + '  ' + 'Test set number:' + str(R_dic['test_len']) + '  '
          + 'Epoch:' + str(R_dic['epochs']))
    print('model name(.pt):' + str(R_dic['model_name']))
    print('=' * 80)

    # ==================== Load data ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join(R_dic['Data_path'], R_dic['train_path'])
    val_path = os.path.join(R_dic['Data_path'], R_dic['val_path'])
    test_path = os.path.join(R_dic['Data_path'], R_dic['test_path'])

    x_train, y_train = \
        Data_NS(train_path, R_dic['train_len'], R_dic['T_in'], R_dic['T'], train_data=True)

    x_val, y_val = \
        Data_NS(val_path, R_dic['val_len'], R_dic['T_in'], R_dic['T'], train_data=False)

    x_test, y_test = \
        Data_NS(test_path, R_dic['test_len'], R_dic['T_in'], R_dic['T'], train_data=False)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train.contiguous(), y_train.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val.contiguous(), y_val.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test.contiguous(), y_test.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)

    # ==================== set the seed ====================
    seed = R_dic['seed']
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # ==================== Load model, optimizer, learning rate scheduler, loss function ====================
    if 'in_dim' not in R_dic:
        R_dic['in_dim'] = 1
    if R_dic['model'] == 'DCNO2d':
        model = DCNO2d(R_dic)

    summary(model, input_size=(R_dic['batch_size'], R_dic['in_dim'], R_dic['res_input'], R_dic['res_input']))
    print('Parameters number:' + str(count_params(model)))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    lr_scheduler = OneCycleLR(optimizer, max_lr=R_dic['max_lr'],
                              div_factor=R_dic['div_factor'],
                              final_div_factor=R_dic['final_div_factor'],
                              pct_start=R_dic['pct_start'],
                              steps_per_epoch=1,
                              epochs=R_dic['epochs'],
                              )
    loss_func = LpLoss(size_average=False)
    epochs = R_dic['epochs']

    loss_train = []
    loss_val = []
    loss_test = []
    lr_history = []
    stop_counter = 0
    best_val_metric = np.inf
    test_by_val = np.inf
    best_val_epoch = None

    Figure_PATH = os.path.join('figures', R_dic['fig_name'])
    lr_name = str('lr_') + R_dic['fig_name']
    lr_PATH = os.path.join('figures', lr_name)

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            train_l2_add, train_l2_full, lr = train_ns(model, train_loader, loss_func, optimizer, lr_scheduler,
                                                       R_dic['train_len'], R_dic['T_in'], R_dic['T'], device=device)
            loss_train.append([train_l2_add, train_l2_full])
            lr_history.append(lr)
            val_l2_add, val_l2_full = test_ns(model, val_loader, loss_func, R_dic['val_len'], R_dic['T'], device=device)
            loss_val.append([val_l2_add, val_l2_full])
            test_l2_add, test_l2_full = test_ns(model, test_loader, loss_func, R_dic['test_len'], R_dic['T'], device=device)
            loss_test.append([test_l2_add, test_l2_full])

            if val_l2_full < best_val_metric:
                best_val_epoch = epoch
                best_val_metric = val_l2_full
                test_by_val = val_l2_full
                stop_counter = 0

                torch.save(model, os.path.join(R_dic['model_save_path'], R_dic['model_name']))

            else:
                stop_counter += 1

            desc = color(f"| Test L2 loss: {test_l2_full:.3e} ", color=Colors.blue)
            desc += color(f"| test by val: {test_by_val:.3e} at epoch {best_val_epoch + 1}", color=Colors.green)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.green)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)
            desc_ep = color("", color=Colors.red)
            desc_ep += color(f"| Train L2 loss : {train_l2_full:.3e} ", color=Colors.red)
            desc_ep += color(f"| Val L2 loss : {val_l2_full:.3e} ", color=Colors.yellow)
            desc_ep += desc
            pbar.set_description(desc_ep)
            pbar.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                test_by_val=test_by_val,
                loss_train=np.asarray(loss_train),
                loss_test=np.asarray(loss_test),
                lr_history=np.asarray(lr_history),
                optimizer_state=optimizer.state_dict()
            )

            save_pickle(result, os.path.join(R_dic['model_save_path'], R_dic['result_name']))
            loss_train_p = np.asarray(loss_train)
            loss_test_p = np.asarray(loss_test)
            x = np.arange(0, epoch + 1, 1)
            plt.style.use('seaborn')
            fig1 = plt.figure()
            plt.semilogy(x, loss_train_p[..., 0], color='red', linestyle="-", linewidth=1, label="train_l2_all")
            plt.semilogy(x, loss_train_p[..., 1], color='red', linestyle="-.", linewidth=1, label="train_l2_ful")
            plt.semilogy(x, loss_test_p[..., 0], color='blue', linestyle="-", linewidth=1, label="test_l2_all")
            plt.semilogy(x, loss_test_p[..., 1], color='blue', linestyle="-.", linewidth=1, label="test_l2_ful")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(fontsize=15)
            plt.savefig(Figure_PATH)
            plt.close(fig1)
            fig2 = plt.figure()
            lr_history_p = np.asarray(lr_history)
            plt.plot(x, lr_history_p, color='green', linestyle="-", linewidth=1, label="learning rate")
            plt.xlabel("Epochs")
            plt.ylabel("lr")
            plt.legend(fontsize=15)
            plt.savefig(lr_PATH)
            plt.close(fig2)

    print('END')
    print('Test error:' + str(test_by_val))
    return result

