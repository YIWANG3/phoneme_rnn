import numpy as np
import os
import pandas as pd
import csv
import Levenshtein
import time
from collections import Counter

RAW_DATA = "ensemble_source"
ENSEMBLE_RESULT = "ensemble_result"

from lib import file_opt

files = file_opt.get_files(RAW_DATA, file_type="csv")
output_file_name = ""
for f in files:
    output_file_name += (f.split("/")[-1]).split(".")[0] + "__"
output_file_name = f'{time.strftime("%m-%d_%H-%M-%S", time.localtime())}__{output_file_name[0:20]}'
print("Save to:" + output_file_name)


def get_result_num():
    sample_csv = files[0]
    csv_file = open(sample_csv)
    dict_reader = csv.DictReader(csv_file)
    result_num = 0
    for _ in dict_reader:
        result_num += 1
    return result_num


result_num = get_result_num()
all_result = []
label = None

for j in range(len(files)):
    f = files[j]
    file_data = pd.read_csv(f, sep=',')
    label = file_data["id"].to_numpy()
    data = file_data["Predicted"].to_numpy()
    all_result.append(data)

all_result = np.array(all_result)
all_result = all_result.T


def get_lowest_sum_dist_str(strs):
    distances = {}
    for i in range(len(strs)):
        dist_sum = 0
        for j in range(len(strs)):
            if i != j:
                dist_sum += Levenshtein.distance(strs[i], strs[j])
        distances[strs[i]] = dist_sum
    dist_arr = sorted(distances.items(), key=lambda item: item[1])
    return dist_arr[0][0]


def ensemble(data):
    result = []
    for i in range(data.shape[0]):
        result.append(get_lowest_sum_dist_str(data[i]))
    return result


def extract_cur_chars(row, index):
    res = []
    for word in row:
        if len(word) > index:
            res.append(word[index])
    return res


def ensemble_stage_two(ensembled, raw_data):
    result = []
    same_len_count = 0
    for i in range(len(ensembled)):
        is_same_len = True
        cur_row = raw_data[i]
        cur_len = len(ensembled[i])
        for line in cur_row:
            if len(line) != cur_len:
                is_same_len = False
                break
        if is_same_len:
            cur_str = ""
            same_len_count += 1
            for l in range(cur_len):
                chars = extract_cur_chars(cur_row, l)
                count_most = Counter(chars).most_common(1)
                cur_str += count_most[0][0]
            result.append(cur_str)
        else:
            result.append(ensembled[i])
    print("same_len_count:", same_len_count)
    return result


def calculate_same_rate(ensembled, raw_data):
    picks = []
    for i in range(len(ensembled)):
        cur_ensemble = ensembled[i]
        cur_row = raw_data[i]
        for p in range(len(cur_row)):
            if cur_ensemble == cur_row[p]:
                picks.append(p)
                continue
    file_count = np.bincount(np.array(picks))
    summary = [f'{i}\t{file_count[i]}\t{files[i].split("/")[-1]}' for i in range(len(files))]
    print('\n'.join(summary))


res = ensemble(all_result)
calculate_same_rate(res, all_result)
res_stage_2 = ensemble_stage_two(res, all_result)

file_opt.export_to_csv(label, "id", res, "Predicted", os.path.join(ENSEMBLE_RESULT, output_file_name + ".csv"))
# file_opt.export_to_csv(label, "id", res_stage_2, "Predicted", os.path.join(ENSEMBLE_RESULT, output_file_name + "_s2.csv"))
