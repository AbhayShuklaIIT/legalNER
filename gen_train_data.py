from utility import *
import pandas as pd
import glob
import random
import os
import shutil


def split_train_test(train, test):
    print("Making train and test")
#     files = glob.glob("./all_files/*.xlsx")
#     random.Random(4).shuffle(files)
#     train_size = int(len(files)*ratio)
#     train, test = files[:train_size], files[train_size:]
    shutil.rmtree("./train/")
    os.mkdir("./train/")
    for i in train:
        name = i[i.rfind("/")+1:]
#         print(name)
        df = pd.read_excel(i,index_col = 0, na_filter = False)
        df.to_excel("./train/"+name)
    shutil.rmtree("./test/")
    os.mkdir("./test/")
    for i in test:
        name = i[i.rfind("/")+1:]
#         print(name)
        df = pd.read_excel(i,index_col = 0, na_filter = False)
        df.to_excel("./test/"+name)

def get_x_y_per_doc(NEs, contexts, roles, label_to_int_dict):
    x = []
    y_f = []
    for ne, context, role in zip(NEs, contexts, roles):
        if len("".join(context))<10:continue
        role_l_int = []
        for i in role:
            if i in label_to_int_dict.keys():
                role_l_int.append(label_to_int_dict[i])
        y_ex = []
        for i in range(len(label_to_int_dict)):
            if i in role_l_int:
                y_ex.append(1)
            else:
                y_ex.append(0)
        if sum(y_ex) == 0:
                # print("SUM Y is 0", ne, role)
                continue
        for c in context:
            x.append(get_x_c_s(ne,c))
            y_f.append(y_ex) 
    return x, y_f

def get_x_y(label_to_int_dict, test_req = 1, combine_ctx = 0):
    NEs, contexts, roles, spacy_labels, NEs_test, contexts_test, roles_test, spacy_labels_test = get_data(combine_ctx = combine_ctx, test_req = test_req)
    roles = correct_roles(roles)
    roles_test = correct_roles(roles_test)
    x, y_f = get_x_y_per_doc(NEs, contexts, roles, label_to_int_dict)
    if test_req == 1:
        x_test, y_f_test = get_x_y_per_doc(NEs_test, contexts_test, roles_test, label_to_int_dict)
    else:
        x_test, y_f_test = [], []
    
    return (x), (y_f), (x_test), (y_f_test), label_to_int_dict

def gen_training_data_cls(train_file ,test_req = 0, combine_ctx = 0):
    print("Combined", combine_ctx)
    label_to_int_dict = get_label_to_int_dict()
    print("Label to int dict", label_to_int_dict)
    x_train, y_train, x_test, y_test, label_to_int_dict = get_x_y(label_to_int_dict, test_req = test_req, combine_ctx = combine_ctx)
    df_dict = {"x_train" : x_train, "y_train" : y_train}
    df = pd.DataFrame.from_dict(df_dict)
    print("Saving data to " + train_file + ".pkl")
    df.to_pickle(train_file + ".pkl")