# -*- coding: utf-8 -*-
from module import common_process as common_p
from module import class_process as class_p

import json
import argparse
import os
import random
random.seed(777)

import time

def main():
    ##############################################################################################
    # initial parameter
    ##############################################################################################
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess raw json data')

    parser.add_argument('--load_path', required=True, metavar="/path/to/annotation/", help='Path to annotation file (.json)')
    parser.add_argument('--save_path', required=True, metavar="/path/to/save_root/", help='Path to dataset for train and test model (.json)')
    parser.add_argument('--process_mode', required=True, help='select segmentation or bounding_box')

    args = parser.parse_args()
    print("load_path : ", args.load_path)
    print("save_path : ", args.save_path)
    print("process_mode : ", args.process_mode)

    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    PROPROCESS_MODE = args.process_mode

    ##############################################################################################
    # define path and file config
    ##############################################################################################
    print('****************************************************************************')
    print('main', '\n',)
    ####### load json
    json_paths_list = common_p.find_json_paths(LOAD_PATH)
    data_dict = common_p.load_multi_json(json_paths_list)

    ####### preprocess
    data_dict = common_p.reset_image_id(data_dict)
    data_mode_dict = common_p.preprocess_annotation(PROPROCESS_MODE, data_dict)
    data_mode_dict = split_unit_image(data_mode_dict)

    ####### write
    write_coco_to_splited(data_mode_dict, PROPROCESS_MODE, SAVE_PATH)
    

###################################################################################################################
# preprocess (Mask RCNN)
###################################################################################################################
def split_unit_image(input_dict):
    print('*********************************************************************')
    print('split_unit_image :', '\n',
            'input keys :', input_dict.keys(), '\n',)
    ####### initial
    output_dict = {}
    num_train_image = 0
    num_val_image = 0
    num_test_image = 0

    ####### repeat for json_file
    for idx_video, json_file_name in enumerate(input_dict):
        ####### initial
        video_dict = input_dict[json_file_name]
        train_annotations = []
        val_annotations = []
        test_annotations = []
        
        ####### parent_attribute
        categories_dict = video_dict['categories']
        images_list = video_dict['images']
        annotations_list = video_dict['annotations']

        ####### repeat to read attributes in annotations
        temp_unit_image = {}
        for idx_anno, annotations_dict in enumerate(annotations_list):
            image_id = annotations_dict['image_id']

            ####### unit image
            if temp_unit_image.get(image_id):
                temp_unit_image[image_id].append(annotations_dict)
            else:
                temp_unit_image[image_id] = []
                temp_unit_image[image_id].append(annotations_dict)

        image_annotations_list = []
        for image_id in temp_unit_image:
            image_annotations_list.append(temp_unit_image[image_id])

        ####### split
        num_images = len(image_annotations_list)
        random.shuffle(image_annotations_list)
        idx_01 = int(num_images*0.8)
        idx_02 = int(num_images*0.9)

        num_train_image += len(image_annotations_list[:idx_01])
        num_val_image += len(image_annotations_list[idx_01:idx_02])
        num_test_image += len(image_annotations_list[idx_02:])

        for annotation in image_annotations_list[:idx_01]:
            train_annotations.extend(annotation)
        for annotation in image_annotations_list[idx_01:idx_02]:
            val_annotations.extend(annotation)
        for annotation in image_annotations_list[idx_02:]:
            test_annotations.extend(annotation)
        
        if idx_video == 0:
            output_dict['categories'] = categories_dict
            output_dict['images'] = images_list
            output_dict['annotations'] = {}
            output_dict['annotations']['train'] = train_annotations
            output_dict['annotations']['val'] = val_annotations
            output_dict['annotations']['test'] = test_annotations
        else:
            output_dict['images'].extend(images_list)
            output_dict['annotations']['train'].extend(train_annotations)
            output_dict['annotations']['val'].extend(val_annotations)
            output_dict['annotations']['test'].extend(test_annotations)

    ####### shuffle for avoiding overfitting
    random.shuffle(output_dict['annotations']['train'])
    random.shuffle(output_dict['annotations']['val'])
    random.shuffle(output_dict['annotations']['test'])

    print('train : image_num :', num_train_image, '\n',
            'val : image_num :', num_val_image, '\n',
            'test : image_num :', num_test_image, '\n',
            'sum : image_num :', num_train_image + num_val_image + num_test_image, '\n',)

    print('train : annotations_num :', len(output_dict['annotations']['train']), '\n',
            'val : annotations_num :', len(output_dict['annotations']['val']), '\n',
            'test : annotations_num :', len(output_dict['annotations']['test']), '\n',)
        
    print('output keys :', output_dict.keys(), '\n',)

    return output_dict


###################################################################################################################
# write json (coco format to train dataset)
###################################################################################################################    
def write_coco_to_splited(input_dict, mode, output_root_path):
    print('*********************************************************************')
    print('write_coco_splited :', '\n',
            'input keys :', input_dict.keys(), '\n',)
    ####### path
    train_file_name = mode + '_train.json'
    val_file_name = mode + '_val.json'
    test_file_name = mode + '_test.json'

    train_path = os.path.join(output_root_path, train_file_name)
    val_path = os.path.join(output_root_path, val_file_name)
    test_path = os.path.join(output_root_path, test_file_name)

    common_p.check_to_exist_file(5, train_path)

    ####### data
    train_dict = {'categories' : input_dict['categories'],
                    'images' : input_dict['images'],
                    'annotations' : input_dict['annotations']['train']}

    val_dict = {'categories' : input_dict['categories'],
                    'images' : input_dict['images'],
                    'annotations' : input_dict['annotations']['val']}

    test_dict = {'categories' : input_dict['categories'],
                    'images' : input_dict['images'],
                    'annotations' : input_dict['annotations']['test']}

    with open(train_path, 'w') as fp:
        json.dump(train_dict, fp)
    with open(val_path, 'w') as fp:
        json.dump(val_dict, fp)
    with open(test_path, 'w') as fp:
        json.dump(test_dict, fp)

    num_train_data = len(train_dict['annotations'])
    num_val_data = len(val_dict['annotations'])
    num_test_data = len(test_dict['annotations'])

    print('***************************************************************')
    print('train anno count :', num_train_data)
    print('val anno count :', num_val_data)
    print('test anno count:', num_test_data)


###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()
