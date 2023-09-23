import torch
from train import *

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
os.path.join(SRC_ROOT, 'data')
MODEL_PATH = os.path.join(SRC_ROOT, 'models')

def test_model(R_dic):
    R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    R_dic['res_output'] = int(
        (R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_stride'] + 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(MODEL_PATH, R_dic['modelname'])
    model = torch.load(model_path)
    model = model.to(device)

    train_path = os.path.join(DATA_PATH, R_dic['train_path'])
    test_path = os.path.join(DATA_PATH, R_dic['test_path'])

    x_train, y_train, x_normalizer, y_normalizer = \
        Data_load(train_path, R_dic['train_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], inv_problem=R_dic['inv_problem'], noise=R_dic['noise'], train_data=True)
    x_test, y_test, _, _ = \
        Data_load(test_path, R_dic['test_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], xnormalizer=x_normalizer, inv_problem=R_dic['inv_problem'],
                  noise=R_dic['noise'], train_data=False)

    R_dic['y_norm'] = y_normalizer

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test.contiguous(), y_test.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)

    # ------------------------------compute predict solution----------------------------------------------
    s1 = R_dic['res_output']
    s2 = R_dic['res_input']
    z_true, z = torch.zeros(1, s1, s1).to(device), torch.zeros(1, s1, s1).to(device)
    xinput = torch.zeros(1, s2, s2).to(device)
    loss_func = LpLoss(size_average=False)
    l2error = 0.0
    print('Data load is OK! Preparation is ready! Begin to compute!')
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(-1, s1, s1)
            y = y.reshape(-1, s1, s1)
            lossl2= loss_func(out, y)
            l2error += lossl2

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
    scio.savemat(R_dic['savemat_name'], {'input': xinput, 'truth': z_true, 'output': z})

    l2error = l2error / R_dic['test_len']
    print(f'l2error = {l2error}')

def test_NS_model(R_dic):
    R_dic['in_dim'] = R_dic['T_in']
    R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    R_dic['res_output'] = int(
        (R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_stride'] + 1)

    # ==================== Load data ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(MODEL_PATH, R_dic['modelname'])
    model = torch.load(model_path)
    model = model.to(device)

    test_path = os.path.join(DATA_PATH, R_dic['test_path'])
    x_test, y_test = \
        Data_NS(test_path, R_dic['test_len'], R_dic['T_in'], R_dic['T'], train_data=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test.contiguous(), y_test.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)

    loss_func = LpLoss(size_average=False)
    print('Data load is OK! Preparation is ready! Begin to compute!')
    test_l2_add, test_l2_full = test_ns(model, test_loader, loss_func, R_dic['test_len'], R_dic['T'], device=device)
    print(f'l2error = {test_l2_full}')



if __name__ == "__main__":
    # When running the program, only keep the corresponding part and comment out the others.
    # =======================================darcy flow forward res=256=================================================
    R_dic = {}
    R_dic['modelname'] = 'darcy_res256_forward.pt'
    R_dic['savemat_name'] = 'results/darcy_res256_forward.mat'
    R_dic['train_path'] = 'darcy_rough_train.mat'
    R_dic['test_path'] = 'darcy_rough_test.mat'
    R_dic['train_len'] = 1000
    R_dic['test_len'] = 100
    R_dic['inv_problem'] = False
    R_dic['noise'] = 0
    R_dic['xGN'] = True
    R_dic['resolution_datasets'] = 512
    R_dic['subsample_nodes'] = 1
    R_dic['subsample_stride'] = 2
    R_dic['patch_padding'] = 1
    R_dic['patch_size'] = 4
    R_dic['batch_size'] = 8
    test_model(R_dic)
    # ===========================multiscale trigonometric coefficients forward res=256==================================
    # R_dic = {}
    # R_dic['modelname'] = 'gamblet_res256_forward.pt'
    # R_dic['savemat_name'] = 'results/gamblet_res256_forward.mat'
    # R_dic['train_path'] = 'gamblet_train.mat'
    # R_dic['test_path'] = 'gamblet_test.mat'
    # R_dic['train_len'] = 1000
    # R_dic['test_len'] = 100
    # R_dic['inv_problem'] = False
    # R_dic['noise'] = 0
    # R_dic['xGN'] = True
    # R_dic['resolution_datasets'] = 1023
    # R_dic['subsample_nodes'] = 1
    # R_dic['subsample_stride'] = 4
    # R_dic['patch_padding'] = 2
    # R_dic['patch_size'] = 4
    # R_dic['batch_size'] = 8
    # test_model(R_dic)
    # =======================================darcy flow inverse res=256=================================================
    # R_dic = {}
    # R_dic['modelname'] = 'darcy_res256_inverse_noise0.0.pt'
    # R_dic['savemat_name'] = 'results/darcy_res256_inverse_noise0.0.mat'
    # R_dic['train_path'] = 'darcy_rough_train.mat'
    # R_dic['test_path'] = 'darcy_rough_test.mat'
    # R_dic['train_len'] = 1000
    # R_dic['test_len'] = 100
    # R_dic['inv_problem'] = True
    # R_dic['noise'] = 0.0
    # R_dic['xGN'] = True
    # R_dic['resolution_datasets'] = 512
    # R_dic['subsample_nodes'] = 1
    # R_dic['subsample_stride'] = 2
    # R_dic['patch_padding'] = 1
    # R_dic['patch_size'] = 4
    # R_dic['batch_size'] = 8
    # print('noise = 0.0:')
    # test_model(R_dic)
    #
    # R_dic = {}
    # R_dic['modelname'] = 'darcy_res256_inverse_noise0.1.pt'
    # R_dic['savemat_name'] = 'results/darcy_res256_inverse_noise0.1.mat'
    # R_dic['train_path'] = 'darcy_rough_train.mat'
    # R_dic['test_path'] = 'darcy_rough_test.mat'
    # R_dic['train_len'] = 1000
    # R_dic['test_len'] = 100
    # R_dic['inv_problem'] = True
    # R_dic['noise'] = 0.1
    # R_dic['xGN'] = True
    # R_dic['resolution_datasets'] = 512
    # R_dic['subsample_nodes'] = 1
    # R_dic['subsample_stride'] = 2
    # R_dic['patch_padding'] = 1
    # R_dic['patch_size'] = 4
    # R_dic['batch_size'] = 8
    # print('noise = 0.1:')
    # test_model(R_dic)
    # ===========================multiscale trigonometric coefficients inverse res=256==================================
    # R_dic = {}
    # R_dic['modelname'] = 'gamblet_res256_inverse_noise0.0.pt'
    # R_dic['savemat_name'] = 'results/gamblet_res256_inverse_noise0.0.mat'
    # R_dic['train_path'] = 'gamblet_train.mat'
    # R_dic['test_path'] = 'gamblet_test.mat'
    # R_dic['train_len'] = 1000
    # R_dic['test_len'] = 100
    # R_dic['inv_problem'] = True
    # R_dic['noise'] = 0
    # R_dic['xGN'] = True
    # R_dic['resolution_datasets'] = 1023
    # R_dic['subsample_nodes'] = 1
    # R_dic['subsample_stride'] = 4
    # R_dic['patch_padding'] = 2
    # R_dic['patch_size'] = 4
    # R_dic['batch_size'] = 8
    # print('noise = 0.0:')
    # test_model(R_dic)
    #
    # R_dic = {}
    # R_dic['modelname'] = 'gamblet_res256_inverse_noise0.1.pt'
    # R_dic['savemat_name'] = 'results/gamblet_res256_inverse_noise0.1.mat'
    # R_dic['train_path'] = 'gamblet_train.mat'
    # R_dic['test_path'] = 'gamblet_test.mat'
    # R_dic['train_len'] = 1000
    # R_dic['test_len'] = 100
    # R_dic['inv_problem'] = True
    # R_dic['noise'] = 0.1
    # R_dic['xGN'] = True
    # R_dic['resolution_datasets'] = 1023
    # R_dic['subsample_nodes'] = 1
    # R_dic['subsample_stride'] = 4
    # R_dic['patch_padding'] = 2
    # R_dic['patch_size'] = 4
    # R_dic['batch_size'] = 8
    # print('noise = 0.1:')
    # test_model(R_dic)
    # ===================================Navier-Stokes, v=1e-6, T_in=10, T=15===========================================
    # R_dic = {}
    # R_dic['modelname'] = 'NS64_1e-6_T15.pt'
    # R_dic['train_path'] = 'NS64_1e-6_T15_train.mat'
    # R_dic['test_path'] = 'NS64_1e-6_T15_test.mat'
    # R_dic['T_in'] = 6
    # R_dic['T'] = 9
    # R_dic['train_len'] = 5000
    # R_dic['test_len'] = 500
    # R_dic['resolution_datasets'] = 64
    # R_dic['subsample_nodes'] = 1
    # R_dic['subsample_stride'] = 1
    # R_dic['patch_padding'] = 1
    # R_dic['patch_size'] = 3
    # R_dic['batch_size'] = 16
    # test_NS_model(R_dic)