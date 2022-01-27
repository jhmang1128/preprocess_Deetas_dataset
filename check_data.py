'''
    intro
'''
from module import common_process as common_p
from module import class_process as class_p

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
    ##############################################################################################
    # initial parameter
    ##############################################################################################
    work_date_list = ['21_09_17']
    # work_date_list = ['21_10_21']
    # work_date_list = ['21_11_10']
    # work_date_list = ['21_12_02', '21_11_10']
    # work_date_list = ['21_12_30', '21_12_02', '21_11_10']
    # work_date_list = ['22_01_05']

    # check_dataset = 'raw'
    check_dataset = 'mask'
    # check_dataset = 'conv'

    # mode = 'bounding_box'
    mode = 'segmentation'
    # mode = 'static_action'

    check_proc_type = 3  # 0 = seg, 3 = total, 6 = on_off, 9 = status
    
    file_names_dict = {'json' : [
                            'bounding_box_train.json', 'bounding_box_val.json', 'bounding_box_test.json', 
                            'segmentation_train.json', 'segmentation_val.json', 'segmentation_test.json',
                            'static_action_train.json', 'static_action_val.json', 'static_action_test.json'
                            ],
                        'pickle' : [
                            'train.pickle', 'val.pickle', 'test.pickle'
                            ]
                        }

    ##############################################################################################
    # define path and file config
    ##############################################################################################
    print('****************************************************************************')
    print('main', '\n',)
    ####### work space
    home_path = os.path.expanduser('~')
    Deetas_root_path = os.path.join(home_path, 'maeng_space/dataset_2021/Deetas')

    ####### input path
    raw_json_root_path = os.path.join(Deetas_root_path, 'data_' + work_date_list[0], 'json_raw')
    mask_rcnn_root_path = os.path.join(Deetas_root_path, 'data_' + work_date_list[0], 'json_MaskRCNN')
    conv_lstm_root_path = os.path.join(Deetas_root_path, 'data_' + work_date_list[0], 'json_ConvLSTM')
    image_root_path = os.path.join(Deetas_root_path, 'image_integrated')

    ####### None
    if check_dataset == 'raw':
        json_paths_list = common_p.find_json_paths(work_date_list, Deetas_root_path)
        data_dict = common_p.load_multi_json(json_paths_list)
        data_dict = common_p.reset_image_id(data_dict)
        data_mode_dict = common_p.preprocess_annotation(mode, data_dict)
    
    ####### obj class
    elif check_dataset == 'mask':
        for idx_file in file_names_dict['json'][0 + check_proc_type : 3 + check_proc_type] :
            print(idx_file, '\n',)

            ####### load
            load_full_path = os.path.join(mask_rcnn_root_path, idx_file)
            data_dict = common_p.load_single_json(load_full_path)
            
            ####### print num
            count_obj_image(data_dict)
            count_obj_class(data_dict)

    ####### track
    elif check_dataset == 'conv':
        ####### full path
        total_data_list = []
        for dataset_name in file_names_dict['pickle']:
            load_full_path = os.path.join(mask_rcnn_root_path, dataset_name)

            ####### load
            data_list = common_p.load_single_pickle(load_full_path)
            total_data_list.append(data_list)

        ####### check and compare (act)
        count_act_image(total_data_list)
        # count_total_act_pickle(total_data_list, Num_bbox)
    else:
        print('error : check_dataset : select : raw / mask / conv')


###################################################################################################################
# act - pickle
###################################################################################################################
def count_obj_image(input_dict):
    ### parent attribute
    categories_list = input_dict['categories']
    images_list = input_dict['images']
    annotations_list = input_dict['annotations']
    
    ### check and print
    print('****************************************************************************')
    count_all_image = []
    count_distinct_image = []
    for annotation in annotations_list:
        image_id = annotation['image_id']
        count_all_image.append(image_id)

    
    print('all image : len : ', len(count_all_image),)

    count_distinct_image = list(set(count_all_image))
    print('distinct image : len : ', len(count_distinct_image),)
    print(type(count_distinct_image))


def count_obj_class(input_dict):
    ####### initial
    total_act_class_dict = {'Going':0,
                            'Coming':0,
                            'Crossing':0,
                            'Stopping':0,
                            'Moving':0,
                            'Avoiding':0,
                            'Opening':0,
                            'Closing':0,
                            'On':0,
                            'Off':0}

    ####### parent attribute
    categories_list = input_dict['categories']
    images_list = input_dict['images']
    annotations_list = input_dict['annotations']
    
    ####### check and print
    print('****************************************************************************')
    count_all_image = []
    count_distinct_image = []
    for annotation in annotations_list:
        image_id = annotation['image_id']
        count_all_image.append(image_id)

    
    print('all image : len : ', len(count_all_image),)

    count_distinct_image = list(set(count_all_image))
    print('distinct image : len : ', len(count_distinct_image),)
    print(type(count_distinct_image))


###################################################################################################################
# act (pickle) - num of action
###################################################################################################################
def count_act_image(input_list):
    sum_image_list = []
    for splited_list in input_list:
        image_list = []
        for act_batch in splited_list:
            for annotation in act_batch[1:]:
                image_id = annotation[1]

                image_list.append(image_id)

        sum_image_list.extend(image_list)
        distinct_image_list = list(set(image_list))
        num_image_splited = len(distinct_image_list)
        print('num_image_splited : ', num_image_splited)

    num_sum_image = len(list(set(sum_image_list)))
    print('num_image_sum : ', num_sum_image)


def count_act_annotation(input_dict, total_act_class_dict):
    print('****************************************************************************')
    print('num_action_class of annotation :', '\n')

    ### initial
    sum_count = 0
    action_class_dict = {'Going':0,
                        'Coming':0,
                        'Crossing':0,
                        'Stopping':0,
                        'Moving':0,
                        'Avoiding':0,
                        'Opening':0,
                        'Closing':0,
                        'On':0,
                        'Off':0}

    for annotaions_list in input_dict['action_class']:
        for idx_anno, annotation in enumerate(annotaions_list):
            if idx_anno == 0:
                continue
            action_class = annotation[0]
            
            ### counting
            action_class_dict = class_p.count_action_class(action_class, action_class_dict)

    for act_name in action_class_dict:
        counting = action_class_dict[act_name]
        total_act_class_dict[act_name] += counting
        sum_count += counting
        print('num_' + act_name, ':', counting)
    print('num_sum :', sum_count)

    return total_act_class_dict


###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()