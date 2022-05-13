# 96655
from utility import *
from gen_train_data import *
from train import *
from validate_final_CLS import *

def train_and_test(config):
    print('train files - ' , len(config['train']), config['train'])
    print('test files - ' , len(config['test']), config['test'])
    split_train_test(config['train'], config['test'])
    gen_training_data_cls(config['outpur_data_file'], combine_ctx = config['combined'])
    train_model(data_path = config['outpur_data_file'] + ".pkl", save_model_name = config['model_name']
          , epochs = 10, use_cuda = 1, batch_size = 64)
    p,r,f = print_result_v1(config['model_name'], config['combined'], "cuda")
    j = print_result_v2(config['model_name'], config['combined'], "cuda")
    print('Results',p,r,f,j)
    return [float(p),float(r),float(f),float(j)]

def perform_cross_val():
#     all_files = glob.glob("./all_files/*")
    all_files = get_all_files()
    l = int(len(all_files)/5)
    config1 = {
        'train' : all_files[l:],
        'test' : all_files[:l],
        'outpur_data_file' : 'step1_data_v2',
        'model_name' : 'step1_model_v2.pt',
        'combined' : 0
    }
#     config2 = {
#         'train' : all_files[:l] + all_files[2*l:],
#         'test' : all_files[l:2*l],
#         'outpur_data_file' : 'step2_data',
#         'model_name' : 'step2_model',
#         'combined' : 0
#     }
    
#     config3 = {
#         'train' : all_files[:4*l],
#         'test' : all_files[4*l:],
#         'outpur_data_file' : 'step3_data',
#         'model_name' : 'step3_model.pt',
#         'combined' : 0
#     }
    
    r1 = train_and_test(config1)
#     r2 = train_and_test(config2)
#     r3 = train_and_test(config3)
    
#     print([r1,r2,r3])
    print(r1)
    
perform_cross_val()