from train import *
import argparse

'''
    Table2 Navier-Stokes equation
'''

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_ROOT, 'data')
MODEL_PATH = os.path.join(SRC_ROOT, 'models')
FIG_PATH = os.path.join(os.path.dirname(SRC_ROOT), 'figures')


parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, default=0, help="random seed"
    )
parser.add_argument(
    "--model", type=str, default='DCNO2d', help="Model"
    )
parser.add_argument(
    "--train_path", type=str, default='NS64_1e-4_T30_train.mat', help="training dataset path"
    )
parser.add_argument(
    "--val_path", type=str, default='NS64_1e-4_T30_val.mat', help="validation dataset path"
    )
parser.add_argument(
    "--test_path", type=str, default='NS64_1e-4_T30_test.mat', help="testing dataset path"
    )
parser.add_argument(
    "--T_in", type=int, default=10, help="input time length"
    )
parser.add_argument(
    "--T", type=int, default=20, help="pred time length"
    )
parser.add_argument(
    "--train_len", type=int, default=5000, help="length of training dataset"
    )
parser.add_argument(
    "--val_len", type=int, default=500, help="length of validation dataset"
    )
parser.add_argument(
    "--test_len", type=int, default=500, help="length of testing dataset"
    )
parser.add_argument(
    "--boundary_condition", type=str, default=None, help="boundary condition"
    )
parser.add_argument(
    "--resolution_datasets", type=int, default=64, help="resolution of data sets"
    )
parser.add_argument(
    "--subsample_nodes", type=int, default=1,
    help="res_input = int((resolution_datasets - 1) / subsample_nodes + 1)"
    )
parser.add_argument(
    "--subsample_stride", type=int, default=1, help="res_output = int((resolution_input - patch_size+2*patch_padding) / stride + 1)"
    )
parser.add_argument(
    "--patch_padding", type=int, default=1, help="patch padding in encoder"
    )
parser.add_argument(
    "--patch_size", type=int, default=3, help="patch size in encoder"
    )
parser.add_argument(
    "--dilation", type=list, default=[1, 3], help="conv dilation in MCNN"
    )
parser.add_argument(
    "--FNO_padding", type=int, default=0, help="FNO padding"
    )
parser.add_argument(
    "--feature_dim", type=int, default=32, help="feature dim: project x onto a high dimension"
    )
parser.add_argument(
    "--modes", type=int, default=12, help="number of fourier modes"
    )
parser.add_argument(
    "--num_spectral_layers", type=int, default=3, help="num of spectral layers"
    )
parser.add_argument(
    "--mlp_hidden_dim", type=int, default=128, help="hidden dim in Feedforward(MLP)"
    )
parser.add_argument(
    "--batch_size", type=int, default=16, help="batch size"
    )
parser.add_argument(
    "--epochs", type=int, default=500, help="epochs"
    )
parser.add_argument(
    "--max_lr", type=float, default=0.001, help="max learning rate"
    )
parser.add_argument(
    "--div_factor", type=int, default=2, help="start learning rate=max_lr/div_factor"
    )
parser.add_argument(
    "--final_div_factor", type=int, default=20, help="end learning rate = max_lr/div_factor/final_div_factor"
    )
parser.add_argument(
    "--pct_start", type=float, default=0.1, help="the proportion of the rising part of learning rate"
    )




if __name__ == "__main__":
    R_dic = parser.parse_args()
    R_dic = vars(R_dic)
    R_dic['Data_path'] = DATA_PATH
    R_dic['model_save_path'] = MODEL_PATH

    # Each part corresponds to a column in Table 2. When running the program, only keep the corresponding part and comment out the others.
    # ============================================v=1e-3, T_in=10, T=50=================================================
    R_dic['train_path'] = 'NS64_1e-3_T50_train.mat'
    R_dic['val_path'] = 'NS64_1e-3_T50_val.mat'
    R_dic['test_path'] = 'NS64_1e-3_T50_test.mat'
    R_dic['T_in'] = 10
    R_dic['T'] = 40
    train_NS_model(R_dic)
    print('END')
    # ============================================v=1e-4, T_in=10, T=25=================================================
    # R_dic['train_path'] = 'NS64_1e-4_T30_train.mat'
    # R_dic['val_path'] = 'NS64_1e-4_T30_val.mat'
    # R_dic['test_path'] = 'NS64_1e-4_T30_test.mat'
    # R_dic['T_in'] = 10
    # R_dic['T'] = 15
    # train_NS_model(R_dic)
    # print('END')
    # ============================================v=1e-5, T_in=10, T=20=================================================
    # R_dic['train_path'] = 'NS64_1e-5_T20_train.mat'
    # R_dic['val_path'] = 'NS64_1e-5_T20_val.mat'
    # R_dic['test_path'] = 'NS64_1e-5_T20_test.mat'
    # R_dic['T_in'] = 10
    # R_dic['T'] = 10
    # train_NS_model(R_dic)
    # print('END')
    # ============================================v=1e-6, T_in=10, T=15=================================================
    # R_dic['train_path'] = 'NS64_1e-6_T15_train.mat'
    # R_dic['val_path'] = 'NS64_1e-6_T15_val.mat'
    # R_dic['test_path'] = 'NS64_1e-6_T15_test.mat'
    # R_dic['T_in'] = 6
    # R_dic['T'] = 9
    # train_NS_model(R_dic)
    # print('END')
    # ==================================================================================================================

