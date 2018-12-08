import pickle

###############################################################################################################
### No comments here, these are utilities functions for serializing data printing statistics and latex data ### 
###############################################################################################################

off_role_id = {'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5, 'AA': 6, 'AM-ADV': 7, 'AM-CAU': 8, 'AM-DIR': 9,
                   'AM-DIS': 10, 'AM-EXT': 11, 'AM-LOC': 12, 'AM-MNR': 13, 'AM-MOD': 14, 'AM-NEG': 15, 'AM-PNC': 16, 'AM-PRD': 17,
                   'AM-TMP': 18, 'C-A1': 19, 'C-A2': 20, 'C-AM-CAU': 21, 'C-AM-DIR': 22, 'C-AM-MNR': 23, 'R-A0': 24, 'R-A1': 25,
                   'R-A2': 26, 'R-AM-CAU': 27, 'R-AM-EXT': 28, 'R-AM-LOC': 29, 'R-AM-MNR': 30, 'R-AM-TMP': 31,  'R-AM-PNC':32, 'R-A4': 33, 'NULL': 34}


def serialize(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def add_to_counter_dict(dict, obj):
    if not dict.get(obj):
        dict[obj] = max(dict.values()) + 1

def reverse_dict(di):
    out = dict()
    for k in di:
        out[di[k]] = k
    return out

def confusion_mat(pred_truth):

    id_role = reverse_dict({'NULL': 0, 'AM-LOC': 1, 'AM-TMP': 2, 'A1': 3, 'A0': 4, 'AM-MNR': 5, 'A2': 6, 'C-A1': 7, 'A3': 8, 'A4': 9, 'AM-NEG': 10, 'AM-MOD': 11, 'R-A0': 12, 'AM-DIS': 13, 'AM-EXT': 14, 'AM-ADV': 15, 'AM-PNC': 16, 'R-A1': 17, 'R-AM-TMP': 18, 'AM-DIR': 19, 'R-A2': 20, 'AM-PRD': 21, 'R-AM-PNC': 22, 'C-AM-MNR': 23, 'AM-CAU': 24, 'R-A3': 25, 'R-AM-MNR': 26, 'R-AM-LOC': 27, 'C-A0': 28, 'R-AM-EXT': 29, 'C-AM-TMP': 30, 'C-A2': 31, 'A5': 32, 'AM': 33, 'C-AM-LOC': 34, 'C-A3': 35, 'R-AM-CAU': 36, 'R-A4': 37, 'C-AM-ADV': 38, 'R-AM-ADV': 39, 'C-AM-PNC': 40, 'AM-REC': 41, 'C-AM-DIR': 42, 'AM-PRT': 43, 'AM-TM': 44, 'C-AM-EXT': 45, 'AA': 46, 'C-A4': 47, 'R-AA': 48, 'C-AM-DIS': 49, 'C-AM-NEG': 50, 'C-AM-CAU': 51, 'C-R-AM-TMP': 52, 'R-AM-DIR': 53})

    cm = []
    for i in range(len(off_role_id)):
        ri = []
        for ii in range(len(off_role_id)+1):
            ri.append(0)
        cm.append(ri)


    for pred, truth in pred_truth:
        truth_rolename = id_role[truth]
        pred_rolename = id_role[pred]

        truth_id_o = off_role_id[truth_rolename]
        pred_id_o = off_role_id[pred_rolename]
        cm[truth_id_o][pred_id_o] += 1

    for i, r in enumerate(cm):
        cm[i][len(off_role_id)] = list(off_role_id.keys())[i]

    return cm


def TP_FN_FP_per_role(cm):

    id_role = reverse_dict(off_role_id)

    out = dict()
    for i in range(len(cm)):

        TP = cm[i][i]
        FN = 0
        FP = 0

        for j in range(len(cm[i][:-1])):    # sum the row
            if not j==i:
                FN += cm[i][j]

        for j in range(len(cm)):   # sum the column
            if not j == i:
                FP += cm[j][i]

        #precision = round((TP/(TP+FP) if not (TP+FP) == 0 else TP/0.01)*100)
        #recall = round((TP/(TP+FN) if not (TP+FN) == 0 else TP/0.01)*100)
        #F1 = round((2*precision*recall/(precision+recall))) if not precision+recall == 0 else 0

        precision = TP/(TP+FP) if not (TP+FP) == 0 else TP/0.01
        recall = TP/(TP+FN) if not (TP+FN) == 0 else TP/0.01
        F1 = 2*precision*recall/(precision+recall) if not precision+recall == 0 else 0

        out[id_role[i]] = (TP, FP, FN, precision, recall, F1)

    print(tex_singular_tpecc(out))
    print(tex_PLOT_tpecc(out))

    return out




def tex_PLOT_tpecc(dicto):
    out = ""

    for i in range(3,6):
        for k in dicto:
            out += "(" + k + "," + str(dicto[k][i]) + ") "
        out = out[:-1]
        out += "\n\n\n"
    return out


def tex_singular_tpecc(dicto):
    out = ""
    for i in range(6):
        for k in dicto:
            out += str(dicto[k][i]) + "&"
        out = out[:-1]
        out += "\\\\\n"
    return out



def tex_confusion_mat(cm):
    out = ""
    riga = 0
    colonna = 0

    for r in cm:
        out += r[-1] + "&"

        for c in r[:-1]:
            if riga == colonna:
                out += "\cellcolor{green!20}" + str(c) + "&"
            else:
                out += str(c) + "&"

            colonna += 1
        riga += 1
        colonna = 0

        out = out[:-1]
        out += "\\\\\n"

    print(out)



