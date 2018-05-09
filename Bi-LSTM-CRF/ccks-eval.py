# coding: utf-8
def Result(s, m):
    print(s)
    return m


def ccks2_eval(submission_path, truth_path):
    dict_tru = {}
    dict_sub = {}
    with open(truth_path) as tru_file, open(submission_path) as sub_file:
        for tru_line, sub_line in zip(tru_file, sub_file):
            dict_tru[tru_line[:8]] = tru_line[9:-1]
            dict_sub[sub_line[:8]] = sub_line[9:-1]

    symptom_dict, disease_dict, exam_dict, treatment_dict, body_dict = {}, {}, {}, {}, {}
    symptom_g, disease_g, exam_g, treatment_g, body_g, overall_g = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for row_id in dict_tru:
        if row_id not in dict_sub:
            return Result(-1, "missing id: " + row_id)
        else:
            t_lst = dict_tru[row_id].split(';')[:-1]

            for item in t_lst:
                item = item.split(' ')
                overall_g += 1
                if item[1] == '肌腱' or item[1] == '肌腱反射' or item[1] == '红素':
                    item[0] == item[0] + item.pop(1)
                if item[3] == '症状和体征':
                    symptom_g += 1
                    if row_id not in symptom_dict:
                        symptom_dict[row_id] = []
                    symptom_dict[row_id].append(item[:3])
                elif item[3] == '疾病和诊断':
                    disease_g += 1
                    if row_id not in disease_dict:
                        disease_dict[row_id] = []
                    disease_dict[row_id].append(item[:3])
                elif item[3] == '检查和检验':
                    exam_g += 1
                    if row_id not in exam_dict:
                        exam_dict[row_id] = []
                    exam_dict[row_id].append(item[:3])
                elif item[3] == '治疗':
                    treatment_g += 1
                    if row_id not in treatment_dict:
                        treatment_dict[row_id] = []
                    treatment_dict[row_id].append(item[:3])
                elif item[3] == '身体部位':
                    body_g += 1
                    if row_id not in body_dict:
                        body_dict[row_id] = []
                    body_dict[row_id].append(item[:3])
                else:
                    return Result(-1, "unknown label: "+str(item))

    symptom_s, disease_s, exam_s, treatment_s, body_s, overall_s = 0, 0, 0, 0, 0, 0
    symptom_r, disease_r, exam_r, treatment_r, body_r, overall_r = 0, 0, 0, 0, 0, 0
    predict, predict_symptom, predict_disease, predict_exam, predict_treatment, predict_body = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for row_id in dict_sub:
        if row_id not in dict_tru:
            return Result(-1, "unknown id:" + row_id)

        s_lst = set(dict_sub[row_id].split(";")[:-1])
        predict += len(s_lst)

        for item in s_lst:
            if len(item) == 0:
                continue
            item = item.split(' ')
            if len(item) == 5:
                item[0] = item[0] + ' ' + item[1]
                item[1] = item[2]
                item[2] = item[3]
                item[3] = item[4]
            if len(item) != 4 and len(item) != 5:
                for _item in item:
                    print(_item)
                # print(item)
                return Result(-1, "incorrect format around id: " + str(row_id))
            if item[3] == "身体部位":
                predict_body += 1
                if row_id not in body_dict:
                    continue
                if item[:3] in body_dict[row_id]:
                    body_s += 1
                    overall_s += 1
                    body_r += 1
                    overall_r += 1
                else:
                    for gold in body_dict[row_id]:
                        if max(item[1], gold[1]) <= min(item[2], gold[2]):
                            body_r += 1
                            overall_r += 1
                            break
            elif item[3] == "症状和体征":
                predict_symptom += 1
                if row_id not in symptom_dict:
                    continue
                if item[:3] in symptom_dict[row_id]:
                    symptom_s += 1
                    overall_s += 1
                    symptom_r += 1
                    overall_r += 1
                else:
                    for gold in symptom_dict[row_id]:
                        if max(item[1], gold[1]) <= min(item[2], gold[2]):
                            symptom_r += 1
                            overall_r += 1
                            break
            elif item[3] == "疾病和诊断":
                predict_disease += 1
                if row_id not in disease_dict:
                    continue
                if item[:3] in disease_dict[row_id]:
                    disease_s += 1
                    overall_s += 1
                    disease_r += 1
                    overall_r += 1
                else:
                    for gold in disease_dict[row_id]:
                        if max(item[1], gold[1]) <= min(item[2], gold[2]):
                            disease_r += 1
                            overall_r += 1
                            break
            elif item[3] == "检查和检验":
                predict_exam += 1
                if row_id not in exam_dict:
                    continue
                if item[:3] in exam_dict[row_id]:
                    exam_s += 1
                    overall_s += 1
                    exam_r += 1
                    overall_r += 1
                else:
                    for gold in exam_dict[row_id]:
                        if max(item[1], gold[1]) <= min(item[2], gold[2]):
                            exam_r += 1
                            overall_r += 1
                            break
            elif item[3] == "治疗":
                predict_treatment += 1
                if row_id not in treatment_dict:
                    continue
                if item[:3] in treatment_dict[row_id]:
                    treatment_s += 1
                    overall_s += 1
                    treatment_r += 1
                    overall_r += 1
                else:
                    for gold in treatment_dict[row_id]:
                        if max(item[1], gold[1]) <= min(item[2], gold[2]):
                            treatment_r += 1
                            overall_r += 1
                            break

    precision, recall, f1 = {}, {}, {}

    precision['symptom_s'] = symptom_s / predict_symptom
    precision['disease_s'] = disease_s / predict_disease
    precision['exam_s'] = exam_s / predict_exam
    precision['treatment_s'] = treatment_s / predict_treatment
    precision['body_s'] = body_s / predict_body
    precision['overall_s'] = overall_s / predict
    precision['symptom_r'] = symptom_r / predict_symptom
    precision['disease_r'] = disease_r / predict_disease
    precision['exam_r'] = exam_r / predict_exam
    precision['treatment_r'] = treatment_r / predict_treatment
    precision['body_r'] = body_r / predict_body
    precision['overall_r'] = overall_r / predict
    recall['symptom_s'] = symptom_s / symptom_g
    recall['disease_s'] = disease_s / disease_g
    recall['exam_s'] = exam_s / exam_g
    recall['treatment_s'] = treatment_s / treatment_g
    recall['body_s'] = body_s / body_g
    recall['overall_s'] = overall_s / overall_g
    recall['symptom_r'] = symptom_r / symptom_g
    recall['disease_r'] = disease_r / disease_g
    recall['exam_r'] = exam_r / exam_g
    recall['treatment_r'] = treatment_r / treatment_g
    recall['body_r'] = body_r / body_g
    recall['overall_r'] = overall_r / overall_g

    print(str(precision))
    print(str(recall))

    for item in precision:
        f1[item] = 2 * precision[item] * recall[item] / (precision[item] + recall[item]) \
            if (precision[item] + recall[item]) != 0 else 0

    return Result(f1['overall_s'], str(f1))

if __name__ == "__main__":
    print(ccks2_eval('./result.csv', './final_gold_truth.csv'))
