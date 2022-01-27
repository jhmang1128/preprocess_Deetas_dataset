# -*- coding: utf-8 -*-
'''
    작성 :
        정성모
        맹주형(수정)
    용도 :
        video별로 나누어져있는 annotation파일들을 묶고 train, val, test로 분할
    input :
        Deetas dataset의 annotation 파일 (.json)
    output :
        train, val, test으로 나눠진 annoation 파일 (.json)
'''

import json
import sys
import os
import random

def main():
    ### select function
    num_function = int(sys.argv[1])

    ### define input, output path (.json , Deetas to train / val / test)
    inout_path_dict = {}
    input_root_path = "/home/dblab/maeng_space/dataset/deetas/data_21_10_21/json"
    output_root_path = "./output/json_obj"
    
    inout_path_dict["input"] = input_root_path
    inout_path_dict["output"] = output_root_path

    ### excute (save middle process)
    data_dict = load_multi_json(inout_path_dict)
    data_dict = reset_image_id(data_dict)
    data_dict = convert_coco_to_track_unit(data_dict, num_function)
    data_dict = split_track_unit(data_dict)

    coco_dict = {}
    print(data_dict.keys())
    for splited_key in data_dict:
        coco_dict[splited_key] = convert_track_unit_to_coco(data_dict[splited_key])
    write_splited_json(coco_dict, output_root_path, num_function)


def convert_track_unit_to_coco(input_dict):
    print("*********************************************************************")
    print("convert_track_unit_to_coco \n")

    ### initial output dict
    output_dict = {}

    ### parent_attribute
    categories_dict = input_dict["categories"]
    images_list = input_dict["images"]
    tracks_list = input_dict["track"]

    ### save
    output_dict["categories"] = categories_dict
    output_dict["images"] = images_list
    output_dict["annotations"] = []

    ### preprocess
    for annotations_list in tracks_list:
        output_dict["annotations"].extend(annotations_list)

    print(output_dict.keys())
    
    return output_dict


def split_track_unit(input_dict):
    print("*********************************************************************")
    print("split_track_unit \n")
    
    ### initial output dict
    output_dict = {"train":{}, "val":{}, "test":{}}
    train_dict = output_dict["train"]
    val_dict = output_dict["val"]
    test_dict = output_dict["test"]

    
    train_dict["track"] = []
    val_dict["track"] = []
    test_dict["track"] = []

    train_dict["images"] = []
    val_dict["images"] = []
    test_dict["images"] = []

    ### repeat for video
    for idx_video, video_name in enumerate(input_dict):
        video_dict = input_dict[video_name]

        ### parent_attribute
        categories_dict = video_dict["categories"]
        images_list = video_dict["images"]
        track_list = video_dict["track"]

        ### save category to each dataset
        if idx_video == 0:
            train_dict["categories"] = categories_dict
            val_dict["categories"] = categories_dict
            test_dict["categories"] = categories_dict

        ### save image info to each dataset
        train_dict["images"].extend(images_list)
        val_dict["images"].extend(images_list)
        test_dict["images"].extend(images_list)
        
        ### idx to split
        num_track = len(track_list)
        idx_01 = int(num_track * 0.8)
        idx_02 = int(num_track * 0.9)

        ### track
        random.shuffle(track_list)
        train_track_list = track_list[0 : idx_01]
        val_track_list = track_list[idx_01 : idx_02]
        test_track_list = track_list[idx_02 : ]

        ### save annotation to each dataset
        train_dict["track"].extend(train_track_list)
        val_dict["track"].extend(val_track_list)
        test_dict["track"].extend(test_track_list)

        print("total : ", len(track_list),
                "/ train : ", len(train_track_list),
                "/ val : ", len(val_track_list),
                "/ test : ", len(test_track_list))
    
    return output_dict


def convert_coco_to_track_unit(input_dict, num_function):
    print("*********************************************************************")
    print("convert_coco_to_track_unit \n")
    ### initial output dict
    output_dict = {}
    print(input_dict.keys())
    
    ### repeat for video
    for video_name in input_dict:
        video_dict = input_dict[video_name]
        output_dict[video_name] = {}
        
        ### check annotation count
        num_annotation = 0

        ### parent_attribute
        categories_dict = video_dict["categories"]
        images_list = video_dict["images"]
        annotations_list = video_dict["annotations"]

        

        output_dict[video_name]["categories"] = categories_dict
        output_dict[video_name]["images"] = images_list
        output_dict[video_name]["track"] = []

        ### repeat to read attributes in annotations
        temp_dict = {}
        
        for idx_anno, annotations_dict in enumerate(annotations_list):
            if num_function == 0:
                pass
            elif num_function == 1:
                if not annotations_dict["segmentation"]:
                    continue
            else:
                print("*********************************************************************")
                print("key error :", num_function)
                exit()

            if annotations_dict["attributes"].get("track_id"):
                track_id = annotations_dict["attributes"]["track_id"]

        ### main process
            if temp_dict.get(track_id):
                temp_dict[track_id].append(annotations_dict)
            else:
                temp_dict[track_id] = [annotations_dict]

        for track_id in temp_dict:
            output_dict[video_name]["track"].append(temp_dict[track_id])

        for annotaion_list in output_dict[video_name]["track"]:
            num_annotation += len(annotaion_list)

        print("video_name :", video_name,
                "/ previous_annotations_num :", len(annotations_list),
                "/ after_annotations_num :", num_annotation,
                "\n / track_num :", len(output_dict[video_name]["track"]))

    return output_dict


def load_multi_json(inout_path_dict):
    print("*********************************************************************")
    print("load_multi_json \n")

    input_root_path = inout_path_dict["input"]
    json_files_list = os.listdir(input_root_path)
    output_dict = {}

    for json_file_name in json_files_list:
        json_file = os.path.join(input_root_path, json_file_name)
        json_reader = open(json_file,'r')
        file_line = json_reader.readline()
        try:
            json_data = json.loads(file_line)
            print("load_json :", json_file_name)
        except:
            print("load_json :", json_file_name, ": error !!! error !!! error !!!")
            continue

        ### initial
        output_dict[json_file_name] = json_data

    return output_dict


def reset_image_id(input_dict):
    print("*********************************************************************")
    print("reset_image_id \n")
    ### initial
    idx_image = 0

    ### repeat
    for video_name in input_dict:
        ### read parent_attribute
        video_dict = input_dict[video_name]

        categories_dict = video_dict["categories"]
        images_list = video_dict["images"]
        annotations_list = video_dict["annotations"]

        ### check idx
        num_image = len(images_list)
        range_video_idx = (idx_image, idx_image + num_image)

        ### reset idx_image
        for images_dict in images_list:
            images_dict["id"] += idx_image
            if not range_video_idx[0] <= images_dict["id"] or not images_dict["id"] < range_video_idx[1]:
                print("error : out of range : reset idx_image")
                print("range :", range_video_idx)
                print("idx :", images_dict["id"])
                exit()
        
        for annotaion_dict in annotations_list:
            annotaion_dict["image_id"] += idx_image
            if not range_video_idx[0] <= images_dict["id"] or not images_dict["id"] < range_video_idx[1]:
                print("error : out of range : reset idx_image")
            
        ### add idx of next_video
        idx_image += num_image

        print("video_name :", video_name, "/ images_list", len(images_list))
    

    return input_dict

    
def write_splited_json(data_dict, output_root_path, num_function):
    print("*********************************************************************")
    print("write_splited_json \n")
    ### path
    output_category = ""
    if num_function == 0:
        print("preprocessing (ouput : whole)")
        output_category = "whole_"
    elif num_function == 1:
        print("preprocessing (ouput : segmentation)")
        output_category = "seg_"
    elif num_function == 2:
        print("preprocessing (ouput : not_segmentation)")
        output_category = "not_seg_"
    else :
        print("error : num_function")
        exit()

    train_file_name = output_category + "train.json"
    val_file_name = output_category + "val.json"
    test_file_name = output_category + "test.json"

    train_path = os.path.join(output_root_path, train_file_name)
    val_path = os.path.join(output_root_path, val_file_name)
    test_path = os.path.join(output_root_path, test_file_name)

    ### data
    train_data = data_dict["train"]
    val_data = data_dict["val"]
    test_data = data_dict["test"]

    with open(train_path, 'w') as fp:
        json.dump(train_data, fp)
    with open(val_path, 'w') as fp:
        json.dump(val_data, fp)
    with open(test_path, 'w') as fp:
        json.dump(test_data, fp)

    num_train_data = len(train_data["annotations"])
    num_val_data = len(val_data["annotations"])
    num_test_data = len(test_data["annotations"])

    print("***************************************************************")
    print("train count :", num_train_data)
    print("val count :", num_val_data)
    print("test count:", num_test_data)



main()
