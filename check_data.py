'''
    intro
'''
import sys
import os

import pickle
import json
import numpy as np
import random

from PIL import Image

'''
    main
'''
def main():
    ### define path
    load_root_path = "/home/dblab/maeng_space/dataset/deetas/data_21_10_21"
    image_root_path = os.path.join(load_root_path, "/data_21_10_21/image")
    
    # problem_root_path = os.path.join(load_root_path, "back_up")
    
    load_categories_dict = {"obj":"json_obj", "act":"json_act"}

    file_names_dict = {"json" : ["seg_train.json", "seg_val.json", "seg_test.json", 
                                "whole_train.json", "whole_val.json", "whole_test.json"],
                        "pickle" : ["train.pickle", "val.pickle", "test.pickle"]}

    ### initial
    Num_bbox = 5
    
    ####### check OBJ
    ### full path
    # problem_full_path = os.path.join(problem_root_path, load_obj)

    ### load
    # data_dict = load_single_json(load_full_path)
    # prob_dict = load_single_json(problem_full_path)
                
    ### check and compare (obj)
    # check_input_obj(data_dict, image_root_path)
    # compare_input_obj(data_dict, prob_dict, image_root_path)


    ####### check ACT
    ### full path
    total_data_list = []
    for dataset_name in file_names_dict["pickle"]:
        load_full_path = os.path.join(load_root_path, load_categories_dict["act"], dataset_name)

        ### load
        data_list = load_single_pickle(load_full_path)
        total_data_list.append(data_list)

        ### check and compare (act)
    count_total_act_pickle(total_data_list, Num_bbox)

'''
    count num module
'''
def count_total_act_pickle(input_list, Num_bbox):
    total_sum_count = 0
    total_act_class_dict = {"Going":0,
                        "Coming":0,
                        "Crossing":0,
                        "Stopping":0,
                        "Moving":0,
                        "Avoiding":0,
                        "Opening":0,
                        "Closing":0,
                        "On":0,
                        "Off":0}

    for splited_list in input_list:
        # total_act_class_dict = count_act_pickle(splited_list, total_act_class_dict)
        # total_act_class_dict = count_act_pickle(splited_list, total_act_class_dict)
        total_act_class_dict = count_act_batch(Num_bbox, splited_list, total_act_class_dict)
        

    print("****************************************************************************")
    print("total_num_action_class :", "\n")
    for act_name in total_act_class_dict:
        counting = total_act_class_dict[act_name]
        print("total_num_" + act_name, ":", counting)
        total_sum_count += counting
    print("total_sum :", total_sum_count)


def count_act_batch(Num_bbox, input_list, total_act_class_dict):
    print("****************************************************************************")
    print("num_action_class of batch :", "\n")
    ### initial
    sum_count = 0
    act_class_dict = {"Going":0,
                        "Coming":0,
                        "Crossing":0,
                        "Stopping":0,
                        "Moving":0,
                        "Avoiding":0,
                        "Opening":0,
                        "Closing":0,
                        "On":0,
                        "Off":0}

    ### load pickle data
    for idx_anno, annotations_list in enumerate(input_list):
        object_class = annotations_list[0]
        del annotations_list[0]

        num_annotation = len(annotations_list)
        if num_annotation < 5:
            continue
        
        for idx_anno in range(num_annotation - Num_bbox + 1):
            batch_data = annotations_list[idx_anno:idx_anno+5]

            temp_list = []
            compare_list = []
            
            for annotation in batch_data:
                action_class = annotation[0]
                compare_list.append(action_class)
                # compare_list.append("test")
            compare_list = list(set(compare_list))
            if len(compare_list) > 1:
                continue

            for batch_data in batch_data:
                temp_list.append(batch_data[2])

            ### counting
            action_class = compare_list[0]

            if action_class == "Going":
                act_class_dict["Going"] += 1
            elif action_class == "Coming":
                act_class_dict["Coming"] += 1
            elif action_class == "Crossing":
                act_class_dict["Crossing"] += 1
            elif action_class == "Stopping":
                act_class_dict["Stopping"] += 1
            elif action_class == "Moving":
                act_class_dict["Moving"] += 1
            elif action_class == "Avoiding":
                act_class_dict["Avoiding"] += 1
            elif action_class == "Opening":
                act_class_dict["Opening"] += 1
            elif action_class == "Closing":
                act_class_dict["Closing"] += 1
            elif action_class == "On":
                act_class_dict["On"] += 1
            elif action_class == "Off":
                act_class_dict["Off"] += 1
            else:
                print("error : act_class")
                exit()
    
    for act_name in act_class_dict:
        counting = act_class_dict[act_name]
        total_act_class_dict[act_name] += counting
        sum_count += counting
        print("num_" + act_name, ":", counting)
    print("num_sum :", sum_count)

    return total_act_class_dict


def count_act_pickle(input_list, total_act_class_dict):
    print("****************************************************************************")
    print("num_action_class :", "\n")

    ### initial
    sum_count = 0
    act_class_dict = {"Going":0,
                        "Coming":0,
                        "Crossing":0,
                        "Stopping":0,
                        "Moving":0,
                        "Avoiding":0,
                        "Opening":0,
                        "Closing":0,
                        "On":0,
                        "Off":0}

    for annotaions_list in input_list:
        for idx_anno, annotation in enumerate(annotaions_list):
            if idx_anno == 0:
                continue
            action_class = annotation[0]

            ### counting
            if action_class == "Going":
                act_class_dict["Going"] += 1
            elif action_class == "Coming":
                act_class_dict["Coming"] += 1
            elif action_class == "Crossing":
                act_class_dict["Crossing"] += 1
            elif action_class == "Stopping":
                act_class_dict["Stopping"] += 1
            elif action_class == "Moving":
                act_class_dict["Moving"] += 1
            elif action_class == "Avoiding":
                act_class_dict["Avoiding"] += 1
            elif action_class == "Opening":
                act_class_dict["Opening"] += 1
            elif action_class == "Closing":
                act_class_dict["Closing"] += 1
            elif action_class == "On":
                act_class_dict["On"] += 1
            elif action_class == "Off":
                act_class_dict["Off"] += 1
            else:
                print("error : act_class")
                exit()

    for act_name in act_class_dict:
        counting = act_class_dict[act_name]
        total_act_class_dict[act_name] += counting
        sum_count += counting
        print("num_" + act_name, ":", counting)
    print("num_sum :", sum_count)

    return total_act_class_dict


def count_act_annotation(input_dict, total_act_class_dict):
    print("****************************************************************************")
    print("num_action_class of annotation :", "\n")

    ### initial
    sum_count = 0
    act_class_dict = {"Going":0,
                        "Coming":0,
                        "Crossing":0,
                        "Stopping":0,
                        "Moving":0,
                        "Avoiding":0,
                        "Opening":0,
                        "Closing":0,
                        "On":0,
                        "Off":0}

    for annotaions_list in input_dict["action_class"]:
        for idx_anno, annotation in enumerate(annotaions_list):
            if idx_anno == 0:
                continue
            action_class = annotation[0]

            ### counting
            if action_class == "Going":
                act_class_dict["Going"] += 1
            elif action_class == "Coming":
                act_class_dict["Coming"] += 1
            elif action_class == "Crossing":
                act_class_dict["Crossing"] += 1
            elif action_class == "Stopping":
                act_class_dict["Stopping"] += 1
            elif action_class == "Moving":
                act_class_dict["Moving"] += 1
            elif action_class == "Avoiding":
                act_class_dict["Avoiding"] += 1
            elif action_class == "Opening":
                act_class_dict["Opening"] += 1
            elif action_class == "Closing":
                act_class_dict["Closing"] += 1
            elif action_class == "On":
                act_class_dict["On"] += 1
            elif action_class == "Off":
                act_class_dict["Off"] += 1
            else:
                print("error : act_class")
                exit()

    for act_name in act_class_dict:
        counting = act_class_dict[act_name]
        total_act_class_dict[act_name] += counting
        sum_count += counting
        print("num_" + act_name, ":", counting)
    print("num_sum :", sum_count)

    return total_act_class_dict


'''
    compare
'''
def compare_input_obj(input_dict, prob_dict, image_root_path):
    ### parent attribute
    check_input_obj(input_dict)
    check_input_obj(prob_dict)
    
# def check_input_LSTM(load_full_path):
#     with open(load_full_path, "rb") as reader_pickle:
#         read_data = pickle.load(reader_pickle)

#     read_data

'''
    check sample
'''
def check_sample_pickle(input_list):
    print("****************************************************************************")
    print("sample_data :", "\n")
    print(input_list[0])
    print(input_list[1])
    print(input_list[2])

def check_input_obj(input_dict, image_root_path):
    ### parent attribute
    categories_list = input_dict["categories"]
    images_list = input_dict["images"]
    annotations_list = input_dict["annotations"]
    
    ### check and print
    print("****************************************************************************")
    print("compare raw :", "/home/dblab/maeng_space/dataset/deetas/data_21_10_21/image", "\n")
    check_data = []

    for image_dict in images_list:
        check_data.append(image_dict["id"])

        # image_full_path = os.path.join(image_root_path, image_dict["file_name"])
        # image_full_path = os.path.join(image_root_path, "N-B-P-020_001549.jpg")
        # image_full_path = "/home/dblab/maeng_space/dataset/deetas/data_21_10_21/image/N-B-C-008_000037.jpg"

        # try:
        #     image = Image.open(image_full_path)
        #     print("image_full_path :", image_full_path)
        # except:
        #     print("image load error !!!")
    
    check_data = set(check_data)
    # print(check_data.keys())
    print(check_data)
    print(len(check_data))
    print(type(check_data))

'''
    load module
'''
def load_single_json(load_full_path):
    print("****************************************************************************")
    print("annotation path :", load_full_path, "\n")
    
    ### initial output
    output_dict = {}

    ### read
    json_reader = open(load_full_path,'r')
    file_line = json_reader.readline()
    json_data = json.loads(file_line)

    ### initial
    output_dict = json_data

    return output_dict

def load_single_pickle(load_full_path):
    print("****************************************************************************")
    print("pickle path :", load_full_path, "\n")
    with open(load_full_path, "rb") as reader_pickle:
        read_data = pickle.load(reader_pickle)

    print("****************************************************************************")
    print("num_data :", len(read_data), "\n")
    return read_data

main()

