import json
import pandas as pd
import glob
import random
import nltk
import numpy as np
import re
import re, math
from collections import Counter
nltk.download('punkt')

label_to_int_dict = {
                     'APPELLANT': 0,
                     'JUDGE': 1,
                     'APPELLANT COUNSEL': 2,
                     'RESPONDENT COUNSEL': 3,
                     'RESPONDENT': 4,
                     'COURT': 5,
                     'PRECEDENT': 6,
                     'AUTHORITY': 7,
                     'WITNESS': 8,
                     'OTHER' : 9,
#                      'NOT APPLICABLE':10,
                                         
}

def get_cosine(vec1, vec2):
    # print vec1, vec2
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r'\w+')
    return Counter(WORD.findall(text))

def get_similarity(a, b):
    if "HIGH COURT" in a.upper() or "HIGH COURT" in b.upper():
        return 0
    a = text_to_vector(a.strip().lower())
    b = text_to_vector(b.strip().lower())

    return get_cosine(a, b)

def combine_varients(n, c, r, s, f, combined):
    thres = 0.85
    n_n, c_n, r_n, s_n, f_n = [], [], [], [], []
    done = []
    for n_i_i in range(len(n)):
        if n_i_i in done:continue
        n_i = n[n_i_i]        
        v = []
        for n_j_i in range(len(n)):
            n_j = n[n_j_i]
            if n_i_i == n_j_i:continue
            if get_similarity(n_i, n_j) > thres:
                v.append(n_j_i)
                done.append(n_j_i)
#                 print(n_i," ---",n_j)
        n_n.append(n_i)
        c_i = c[n_i_i]
        for v_i in v:
            c_i = c_i + c[v_i]
        if combined == 1:
            c_i = [". ".join(c_i)]  
        c_n.append(c_i)
        r_n.append(r[n_i_i])
        s_n.append(s[n_i_i])
        f_n.append(f[n_i_i])

    return n_n, c_n, r_n, s_n, f_n

def get_label_to_int_dict():
    return label_to_int_dict

def return_text_file(path):
    # print(path)
    f = open(path)
    data = json.load(f)
    text = ""
    for i in data:
        for j in (data[i]):
            text = text + " ".join(data[i][j])
    # print(data)
    return text


def get_context(ne, text, a):
    prev = " ".join(text[:a].split(" ")[-100:])
    # print(prev)
    prev_sents = nltk.tokenize.sent_tokenize(prev)
    next = " ".join(text[a+len(ne):].split(" ")[:100])
    next_sents = nltk.tokenize.sent_tokenize(next)
    ctx = " ".join(prev_sents[-1:]) + "<NE>" + ne + "</NE>" + " ".join(next_sents[:1])
    # print(ctx)
    return ctx

def remove_nan(df):
  cols = df.columns
  for col in cols:
    df = df[df[col] != np.nan]
    df = df[df[col] != float('nan')]
    df = df[df[col].notna()]
  return df


def generate_training_examples(df, path, combine_ctx = 0):
    NEs = []
    contexts = []
    roles = []
    spacy_labels = []
    df.columns = [i.lower() for i in df.columns]
    cols = ['entities', 'labels', 'frequency', 'variant', 'role']
    df = df[cols]
    df = remove_nan(df)
    r_l = list(df["role"])
    s_l = list(df["labels"])
    nes_l = df["entities"]
    varient_l = df["variant"]
    f_l = list(df["frequency"])
    # roles.extend(list(df["role"]))
    # spacy_labels.extend(list(df["labels"]))
    # NEs.extend(list(df["entities"]))

    text = return_text_file("./jsons/" + path[path.rfind("/")+1:-5] + ".json")
    for i in range(len(nes_l)):
        try:
            ne = nes_l[i]
        except Exception as e:
            print(e)
            continue
        try:
            a_s = [m.start() for m in re.finditer(ne, text)]
        except:
            # print("hi" , end = ", ")
            try:
                a_s = [text.find(ne)]
            except:
                print(ne)
                continue
        context = []
        for a in a_s:
            context.append(get_context(ne, text, a))
        if combine_ctx == 1:
            ctx_temp = ". ".join(context)
            context = []
            context.append(ctx_temp)
        try:
            if len(context) > 20:
                context = context[0:20]
            # ctx = sent2vec_model.embed_sentences(context)
            r_t = r_l[i]
            p_i = int(i)
            patience = 10
            while len(r_t) == 0:
                if patience == -1: break
                try:
                    p_i = int(varient_l[p_i])
                except:
                    p_i = varient_l[p_i].split(",")
                    p_i = int(p_i[0])
                    
                r_t = r_l[p_i]
#                 print(p_i)
                patience = patience  - 1
                

            contexts.append(context)
            roles.append(r_t)
            spacy_labels.append(s_l[i])
            NEs.append(nes_l[i])
            if len(r_t) == 0:
#                 print(path, ne)
                pass
        except Exception as e: 
            print("Error catch 1" , e)
            # print(context2)
            continue
        
    return NEs, contexts, roles, spacy_labels, f_l


def get_data(combine_ctx, test_req = 1):
    # skip = [46]
    train = glob.glob("./train/*.xlsx")
    test = glob.glob("./test/*.xlsx")
    NEs = []
    contexts = []
    roles = []
    spacy_labels = []
    cnt = 0
    for i in train:
        print(cnt, end = ", ")
        print(i)
        try:
            n, c, r, s, f = generate_training_examples(pd.read_excel(i, index_col = 0, na_filter = None), i, combine_ctx)
            n, c, r, s, f = combine_varients(n, c, r, s, f, combine_ctx)
        except Exception as e:
            print(i, "ERROR", e)
            continue
        NEs.extend(n)
        contexts.extend(c)
        roles.extend(r)
        spacy_labels.extend(s)
        cnt = cnt + 1
        
#         break
    if test_req == 0:
        return NEs, contexts, roles, spacy_labels, [], [], [], []
    NEs_test = []
    contexts_test = []
    roles_test = []
    spacy_labels_test = []
    for i in test:
        print(i)
#         if i == './test/87753577.xlsx':continue
        try:
            n, c, r, s, _ = generate_training_examples(pd.read_excel(i, index_col = 0, na_filter = None), i, combine_ctx)
        except:
            print(i, "ERROR")
            continue
        NEs_test.extend(n)
        contexts_test.extend(c)
        roles_test.extend(r)
        spacy_labels_test.extend(s)
        cnt = cnt + 1
        print(cnt, end = ", ")
#         break
    
    return NEs, contexts, roles, spacy_labels, NEs_test, contexts_test, roles_test, spacy_labels_test

oth = ["ACQ", "ACC", 'CONV', "VIC", 'PLACE', 'VICTIM', 'Victim', 'Other', 'other', 'POL', 'MEDICAL', "O", "POLICE", "NA"]
# na_and_oth = oth + ['NA']
skip = [""]
jud = ["JUDGE(CC)", "JUDGE(LC)"]
def get_label(label):
    label = label.replace(" ", "")
    label = label.upper()
#     print(label)
    if label in skip:return ""
    # print(label)
    if label[0:4] == "PREC" or label == "STAT" or label == "STATUTE":
        return "PRECEDENT"
    if label == "A.COUNSEL" or label == 'PETITIONER':
        return "APPELLANT COUNSEL"
    if label == "ACOUNSEL":
        return "APPELLANT COUNSEL"
    if label == "R.COUNSEL":
        return "RESPONDENT COUNSEL"
    if label == "O":
        return "OTHER"
    if label == "APP":
        return "APPELLANT"
    if label == "RESP":
        return "RESPONDENT"
    if label == "AUTH" or "AUTHORITY" in label:
        return "AUTHORITY"
    if label in ["D.WITNESS", "P.WITNESS"] or "WITNESS" in label:
        return "WITNESS"
    if label in jud:
        return "JUDGE"
    if "JUDGE" in label:
        return "JUDGE"
    if "COURT" in label:
        return "COURT"
    if "COUNSEL" in label and "APPELLANT" in label:
        return "APPELLANT COUNSEL"
    if "COUNSEL" in label and "RESPONDENT" in label:
        return "RESPONDENT COUNSEL"
    if "APPELLANT" in label:
        return "APPELLANT"
    if "RESPONDENT" in label:
        return "RESPONDENT"
    if "COUNSEL" in label:
        return "APPELLANT COUNSEL"
#     if label in "NA":
#         return "NOT APPLICABLE"
    for i in oth:
        if i in label:
            return "OTHER"
    return label

def correct_roles(roles):
    c_roles = []
    for rs in roles:
        try:
            crs = []
            for r in rs.split(","):
                # print(r)
                v = get_label(r)
                if v.replace(" ", "") == "" or v == "OTHER":continue
                crs.append(v)
                break
            if len(crs) == 0 and len(rs.split(",")) != 0:crs.append("OTHER")
            c_roles.append(crs)
        # except:
        except Exception as e:
            print(e)
            continue
    return c_roles

def get_x_only_c(ne, c):
    return c
def get_x_c_s(ne, c):
    return ne + "[SEP]" + c

def get_all_files():
    return ['./all_files/114254342.xlsx', './all_files/160214.xlsx', './all_files/158170320.xlsx', './all_files/1236969.xlsx', './all_files/62816014.xlsx', './all_files/39503956.xlsx', './all_files/75243301.xlsx', './all_files/1243363.xlsx', './all_files/952082.xlsx', './all_files/37849282.xlsx', './all_files/51061623.xlsx', './all_files/135785387.xlsx', './all_files/1681654.xlsx', './all_files/975074.xlsx', './all_files/1161164.xlsx', './all_files/1646640.xlsx', './all_files/C1601209.xlsx', './all_files/491438.xlsx', './all_files/169899114.xlsx', './all_files/14560127.xlsx', './all_files/154384989.xlsx', './all_files/35302553.xlsx', './all_files/83742049.xlsx', './all_files/1404484.xlsx', './all_files/163837917.xlsx', './all_files/60681802.xlsx', './all_files/49086800.xlsx', './all_files/1052174.xlsx', './all_files/1389243.xlsx', './all_files/115833707.xlsx', './all_files/191617837.xlsx', './all_files/373587.xlsx', './all_files/139618288.xlsx', './all_files/444757.xlsx', './all_files/1622258.xlsx', './all_files/97694707.xlsx', './all_files/661155.xlsx', './all_files/1210867.xlsx', './all_files/186002698.xlsx', './all_files/139796748.xlsx', './all_files/186369757.xlsx', './all_files/1145881.xlsx', './all_files/79816183.xlsx', './all_files/66145267.xlsx', './all_files/28534827.xlsx', './all_files/1011232.xlsx', './all_files/1680408.xlsx', './all_files/374386.xlsx', './all_files/178619490.xlsx', './all_files/77536432.xlsx', './all_files/49260341.xlsx', './all_files/808484.xlsx', './all_files/192156229.xlsx', './all_files/1716348.xlsx', './all_files/194515205.xlsx', './all_files/1128835.xlsx', './all_files/1680750.xlsx', './all_files/53410471.xlsx', './all_files/127273457.xlsx', './all_files/195577943.xlsx', './all_files/19436407.xlsx', './all_files/94780858.xlsx', './all_files/111507500.xlsx', './all_files/1714434.xlsx', './all_files/348439.xlsx', './all_files/1405647.xlsx', './all_files/929793.xlsx', './all_files/1572927.xlsx', './all_files/31538746.xlsx', './all_files/124647567.xlsx', './all_files/125586420.xlsx', './all_files/129202572.xlsx', './all_files/C1461784.xlsx', './all_files/144770867.xlsx', './all_files/80182706.xlsx', './all_files/90251163.xlsx', './all_files/C1622258.xlsx', './all_files/48115323.xlsx', './all_files/26450143.xlsx', './all_files/1661941.xlsx', './all_files/10104667.xlsx', './all_files/95803467.xlsx', './all_files/101881480.xlsx', './all_files/13024806.xlsx', './all_files/160467640.xlsx', './all_files/1378557.xlsx', './all_files/136341809.xlsx', './all_files/184940.xlsx', './all_files/1922398.xlsx', './all_files/1855581.xlsx', './all_files/292181.xlsx', './all_files/806212.xlsx', './all_files/124935621.xlsx', './all_files/1890994.xlsx', './all_files/87753577.xlsx', './all_files/190580.xlsx', './all_files/94732417.xlsx', './all_files/450185.xlsx', './all_files/76051345.xlsx', './all_files/1304109.xlsx', './all_files/42681804.xlsx', './all_files/37724412.xlsx', './all_files/95311661.xlsx', './all_files/1348961.xlsx', './all_files/52024378.xlsx', './all_files/166859104.xlsx', './all_files/77319253.xlsx', './all_files/736324.xlsx', './all_files/69972738.xlsx', './all_files/47053425.xlsx', './all_files/1265850.xlsx', './all_files/171479667.xlsx', './all_files/1601209.xlsx', './all_files/146535534.xlsx', './all_files/110007179.xlsx', './all_files/58889922.xlsx', './all_files/156503.xlsx', './all_files/26913217.xlsx', './all_files/182540940.xlsx', './all_files/1030541.xlsx', './all_files/160722490.xlsx', './all_files/1373037.xlsx', './all_files/189745935.xlsx', './all_files/86495747.xlsx', './all_files/115651329.xlsx', './all_files/88531418.xlsx', './all_files/1246150.xlsx', './all_files/1670685.xlsx', './all_files/C1210867.xlsx', './all_files/195489804.xlsx', './all_files/1305957.xlsx', './all_files/756812.xlsx', './all_files/42020425.xlsx', './all_files/121357872.xlsx', './all_files/1461784.xlsx', './all_files/1324536.xlsx', './all_files/445635.xlsx', './all_files/118956979.xlsx', './all_files/1992267.xlsx', './all_files/1063151.xlsx', './all_files/101482639.xlsx', './all_files/91090158.xlsx', './all_files/764779.xlsx', './all_files/20392876.xlsx', './all_files/1620518.xlsx']


