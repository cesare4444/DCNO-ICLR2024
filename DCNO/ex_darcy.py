from train import *
import argparse

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/dssg/home/acct-nislz/nislz/xubo/MC/data'
# DATA_PATH = 'E:/SJTU/torch_project/MC/data'
MODEL_PATH = os.path.join(SRC_ROOT, 'models')
FIG_PATH = os.path.join(os.path.dirname(SRC_ROOT), 'figures')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, default=4444, help="random seed"
    )
parser.add_argument(
    "--model", type=str, default='DCNO2d', help="Model"
    )
parser.add_argument(
    "--train_path", type=str, default='darcy_rough_train.mat', help="training dataset path"
    )
parser.add_argument(
    "--val_path", type=str, default='darcy_rough_val.mat', help="validation dataset path"
    )
parser.add_argument(
    "--test_path", type=str, default='darcy_rough_test.mat', help="testing dataset path"
    )
parser.add_argument(
    "--train_len", type=int, default=1000, help="length of training dataset"
    )
parser.add_argument(
    "--val_len", type=int, default=100, help="length of validation dataset"
    )
parser.add_argument(
    "--test_len", type=int, default=100, help="length of testing dataset"
    )
parser.add_argument(
    "--inv_problem", type=str, default=False, help="positive problem if False else inverse problem "
    )
parser.add_argument(
    "--noise", type=float, default=0.0, help="add Gaussian noise on the inverse problem"
    )
parser.add_argument(
    "--losstype", type=str, default='H1', help="loss function during training"
    )
parser.add_argument(
    "--H1loss_modes", type=int, default=23, help="control the modes of the loss in fourier domain"
    )
parser.add_argument(
    "--xGN", type=str, default=True,
    help="whether normalize coefficient in positive problem(solution in inverse problem)"
    )
parser.add_argument(
    "--resolution_datasets", type=int, default=512, help="resolution of data sets"
    )
parser.add_argument(
    "--subsample_nodes", type=int, default=1,
    help="res_input = int((resolution_datasets - 1) / subsample_nodes + 1)"
    )
parser.add_argument(
    "--subsample_stride", type=int, default=2, help="res_output = int((resolution_input - patch_size+2*patch_padding) / stride + 1)"
    )
parser.add_argument(
    "--patch_padding", type=int, default=1, help="patch padding in encoder"
    )
parser.add_argument(
    "--patch_size", type=int, default=4, help="patch size in encoder"
    )
parser.add_argument(
    "--dilation", type=list, default=[1,3,9], help="conv dilation in MCNN"
    )
parser.add_argument(
    "--FNO_padding", type=int, default=5, help="FNO padding"
    )
parser.add_argument(
    "--feature_dim", type=int, default=24, help="feature dim: project x onto a high dimension"
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
    "--batch_size", type=int, default=8, help="batch size"
    )
parser.add_argument(
    "--epochs", type=int, default=500, help="epochs"
    )
parser.add_argument(
    "--max_lr", type=float, default=8e-4, help="max learning rate"
    )
parser.add_argument(
    "--div_factor", type=int, default=4, help="start learning rate=max_lr/div_factor"
    )
parser.add_argument(
    "--final_div_factor", type=int, default=8, help="end learning rate = max_lr/div_factor/final_div_factor"
    )
parser.add_argument(
    "--pct_start", type=float, default=0.2, help="the proportion of the rising part of learning rate"
    )

if __name__ == "__main__":
    R_dic = parser.parse_args()
    R_dic = vars(R_dic)
    R_dic['Data_path'] = DATA_PATH
    R_dic['model_save_path'] = MODEL_PATH
    # =========================================Table1, Frequency loss function==========================================
    # #resolution = 256
    # train_2D_model(R_dic)

    # #resolution = 128
    # print('\n\n\n\n')
    # del R_dic['model_name']
    # R_dic['subsample_stride'] = 4
    # train_2D_model(R_dic)
    # ==========================================Table1, L2 loss=========================================================
    # #resolution = 256
    R_dic['losstype'] = 'L2'
    print('\n\n\n\n')
    if 'model_name' in R_dic:
        del R_dic['model_name']
    train_2D_model(R_dic)

    # #resolution = 128
    print('\n\n\n\n')
    if 'model_name' in R_dic:
        del R_dic['model_name']
    R_dic['subsample_stride'] = 4
    train_2D_model(R_dic)
    # =======================================Table2 Inverse problem=====================================================
    # mode_record= [0, 0.01, 0.05, 0.1]
    # result_record = []
    # R_dic['inv_problem'] = True
    # R_dic['boundary_condition'] = None
    # R_dic['losstype'] = 'L2'
    # for i in mode_record:
    #     if 'model_name' in R_dic:
    #         del R_dic['model_name']
    #     R_dic['noise'] = i
    #     result = train_2D_model(R_dic)
    #     result_record.append(result)
    #     print(result_record)
    # print(mode_record)
    # print(result_record)

    # =============Ablation Study===========
    # R_dic['subsample_nodes'] = 2
    # R_dic['subsample_stride'] = 1
    # R_dic['patch_size'] = 3
    # if 'model_name' in R_dic:
    #     del R_dic['model_name']
    # train_2D_model(R_dic)