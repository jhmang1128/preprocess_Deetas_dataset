# -*- coding: utf-8 -*-
'''
    작성 :
        정성모
        맹주형(수정)
    용도 :
        video, track 단위의 학습 데이터 생성 (train : valiadation : test, 8 : 1 : 1 )
    input :
        Deetas datset의 annotation (.json) 파일
    output :
        video, track 단위의 학습 데이터 생성 (train, validation, test dataset (pickle))
'''

import sys
import os

import pickle
import json

import random

def main():
    ### define json path (Deetas to track_unit)
    json_root_path = "/home/dblab/maeng_space/dataset/deetas/data_21_10_21/json"

    ### define out path (Deetas to track_unit)
    output_root_path = "./output"
    output_file_name = "track_unit.json"
    output_file_path = os.path.join(output_root_path, output_file_name)

    ### excute (save middle process)
    track_dict = preprocess_to_track_unit(json_root_path)
    # write_json(output_file_path, track_dict)


    ### define json path (track_unit.json to pickle_list)
    # track_root_path = "./output"
    

    ### define out path (track_unit.json to pickle_list)
    output_root_path = "./output"
    

    ### excute
    output_list = create_input_LSTM(track_dict, 0)
    # write_pickle(output_list, output_root_path)
    

def preprocess_to_track_unit(json_root_path):
    ### read file
    json_file_list = os.listdir(json_root_path)

    ### initial
    output_dict = {}

    ### repeat for json_file
    for json_file_name in json_file_list:
        json_file = os.path.join(json_root_path, json_file_name)
        print("preprocessing :", json_file)
        json_reader = open(json_file,'r')
        file_line = json_reader.readline()
        try:
            json_data = json.loads(file_line)
        except:
            print("error")
            continue

        ### initial
        output_dict[json_file_name] = []
        
        ### parent_attribute
        categories_dict = json_data["categories"]
        images_dict = json_data["images"]
        annotations_array = json_data["annotations"]

        ### repeat to read attributes in annotations
        temp_dict = {}
        for annotations_dict in annotations_array:
            if annotations_dict["attributes"].get("track_id"):
                temp_anno_dict = {}
                image_id = annotations_dict["image_id"]
                segmentation = annotations_dict["segmentation"]
                bbox = annotations_dict["bbox"]
                track_id = annotations_dict["attributes"]["track_id"]
                status = annotations_dict["attributes"]["Status"]
                category_id = annotations_dict["category_id"]

        ### main process
            if temp_dict.get(track_id):
                temp_dict[track_id]["annotations"].append({"image_id":image_id, "bbox":bbox, "segmentation":segmentation, "Status":status})
            else:
                temp_dict[track_id] = {"id":track_id, "category_id":category_id,
                                    "annotations":[{"image_id":image_id, "bbox":bbox, "segmentation":segmentation, "Status":status}]}

        for track_id in temp_dict:
            output_dict[json_file_name].append(temp_dict[track_id])

    return output_dict


def create_input_LSTM(videos_track_dict, read_mode):
    ### initial (for output)
    output_list = []
    train_data = []
    val_data = []
    test_data = []

    ### read track_dict
    for track_key in videos_track_dict:
        tracks_list = videos_track_dict[track_key]
        temp_tracks_list = []
        for track_dict in tracks_list:
            temp_track_list = []
            category_id = track_dict["category_id"]
            temp_track_list.append(category_id)
            annotations_list = track_dict["annotations"]
            for annotation_dict in annotations_list:
                temp_inner_list = []
                temp_inner_list.append(annotation_dict["Status"])
                temp_inner_list.append(annotation_dict["image_id"])
                temp_inner_list.append(annotation_dict["bbox"])
                temp_track_list.append(temp_inner_list)
            temp_tracks_list.append(temp_track_list)
        
        random.shuffle(temp_tracks_list)

        ### data divide
        num_tracks = len(temp_tracks_list)
        idx_val = int(num_tracks*0.8)
        idx_test = int(num_tracks*0.9)
        
        train_data.extend(temp_tracks_list[:idx_val])
        val_data.extend(temp_tracks_list[idx_val:idx_test])
        test_data.extend(temp_tracks_list[idx_test:])
        
    output_list.append(train_data)
    output_list.append(val_data)
    output_list.append(test_data)

    print("*************************************************************")
    print("output_list : ", len(output_list))
    print("train_num : ", len(train_data))
    print("val_num : ", len(val_data))
    print("test_num : ", len(test_data))

    return output_list


def write_pickle(output_list, output_root_path):
    ### define output path
    output_file_name_list = ["train.pickle", "val.pickle", "test.pickle"]
    train_path = os.path.join(output_root_path, output_file_name_list[0])
    val_path = os.path.join(output_root_path, output_file_name_list[1])
    test_path = os.path.join(output_root_path, output_file_name_list[2])

    train_data = output_list[0]
    val_data = output_list[1]
    test_data = output_list[2]

    train_writer = open(train_path, 'wb')
    pickle.dump(train_data, train_writer)
    train_writer.close()

    val_writer = open(val_path, 'wb')
    pickle.dump(val_data, val_writer)
    val_writer.close()

    test_writer = open(test_path, 'wb')
    pickle.dump(test_data, test_writer)
    test_writer.close()


def write_json(output_file_path, json_data):
    fp = open(output_file_path, 'w')
    json.dump(json_data, fp)


main()