# -*- coding: utf-8 -*-
'''
    작성 :
        맹주형
    용도 :
        Deetas dataset에 대한 공통 적인 전처리 과정
        json 파일을 읽어 딕셔너리 자료형으로 반환
'''
from module import class_process as class_p
# import class_process as class_p

import os
import random
import time

import pickle
import json
import numpy as np
import h5py


###################################################################################################################
# reset image id
###################################################################################################################
def reset_image_id(input_dict):
    print('*********************************************************************')
    print('reset_image_id :', '\n',
            'input keys :', input_dict.keys(), '\n',)
    ####### initial
    idx_image = 0
    num_video_split = 100
    num_total_image = 0

    ####### repeat
    for video_name in input_dict:
        check_error_file = False

        ####### read parent_attribute
        video_dict = input_dict[video_name]

        idx_cut = video_name.rfind('.')
        video_str = video_name[:idx_cut]

        categories_dict = video_dict['categories']
        images_list = video_dict['images']
        annotations_list = video_dict['annotations']

        ####### check idx
        num_image = len(images_list)
        range_video_idx = (idx_image, idx_image + num_image)

        
        ####### check_image_file_name
        for image_single_dict in images_list:
            if not video_str in image_single_dict['file_name']:
                check_error_file = True

        if not check_error_file:
            continue

        ####### reset idx_image in image
        for images_dict in images_list:
            images_dict['id'] += idx_image
            ####### check range
            if not range_video_idx[0] <= images_dict['id'] or not images_dict['id'] <= range_video_idx[1]:
                print('error : out of range : reset idx_image')
                print('range :', range_video_idx)
                print('idx :', images_dict['id'])
                exit()
        
        ####### reset idx_image in annotation
        for annotaion_dict in annotations_list:
            annotaion_dict['image_id'] += idx_image
            ####### check range
            if not range_video_idx[0] <= annotaion_dict['image_id'] or not annotaion_dict['image_id'] <= range_video_idx[1]:
                print('error : out of range : reset idx_image')
            
        ####### start index of next_video
        idx_image += num_image + num_video_split

        print(video_name, ': image num :', len(images_list))
        num_total_image += len(images_list)

    print('total image num :', num_total_image, '\n',)

    return input_dict


###################################################################################################################
# segmentation
###################################################################################################################
def preprocess_annotation(mode, input_dict):
    print('*********************************************************************')
    print('preprocess_annotation :', '\n',
            'input keys :', input_dict.keys(), '\n',)
    ####### initial
    output_dict = {}
    num_total_annotation = 0
    num_mode_annotation = 0
    num_sum_classes = 0
    image_list_total = []
    image_list_mode = []
    check_num_class_dict = {}
    
    ####### repeat for video
    for video_name in input_dict:
        video_dict = input_dict[video_name]
        output_dict[video_name] = {}

        temp_image_list_total = []
        temp_image_list_mode = []

        ####### category id
        if mode == 'bounding_box' : class_list = video_dict['categories']
        elif mode == 'segmentation' : class_list = class_p.generate_static_except_Button()
        elif mode == 'dynamic_action' : class_list = class_p.generate_dynamic_action()
        elif mode == 'static_action' : class_list = class_p.generate_static_action()
        else:
            print('select wrong mode : check your mode :', mode)
            exit()

        ####### parent attribute
        output_dict[video_name]['categories'] = class_list
        output_dict[video_name]['images'] = video_dict['images']
        output_dict[video_name]['annotations'] = out_annotations_list = []

        annotations_list = video_dict['annotations']

        ####### foreach annotaion
        for annotations_dict in annotations_list:
            temp_image_list_total.append(annotations_dict['image_id'])
            
            ####### mode for object detection
            if mode == 'bounding_box':
                if not annotations_dict['bbox']:
                    continue
                
            ####### mode for maskRCNN
            elif mode == 'segmentation':
                if annotations_dict['category_id'] <= 5 : continue
                if not annotations_dict['bbox'] : continue
                if not annotations_dict['segmentation'] : continue
                
                if 5 < annotations_dict['category_id'] <= 17 : annotations_dict['category_id'] -= 5
                elif 18 <= annotations_dict['category_id'] <= 23 : continue # except Button
                elif annotations_dict['category_id'] == 24 : annotations_dict['category_id'] -= 11
                elif annotations_dict['category_id'] == 25 : continue # except bell
                else:
                    print('error : category_id / in process annotation module')
                    exit()
                
            ####### mode for maskRCNN
            elif mode == 'static_action':
                if annotations_dict['category_id'] <= 5 : continue
                if not annotations_dict['attributes'].get('Status') : continue
                if not annotations_dict['segmentation'] : continue
                if not annotations_dict['bbox'] : continue
                
                annotations_dict = class_p.converted_into_static_action(annotations_dict)
                
                if not annotations_dict:
                    continue
                
            ####### mode for Convolution LSTM
            elif mode == 'dynamic_action':
                if not annotations_dict['category_id'] <= 5 : continue
                if not annotations_dict['attributes'].get('Status') : continue
                if not annotations_dict['bbox'] : continue

                # not now (process class)
                
            if not check_num_class_dict.get(annotations_dict['category_id']):
                check_num_class_dict[annotations_dict['category_id']] = []
                check_num_class_dict[annotations_dict['category_id']].append('')
            else:
                check_num_class_dict[annotations_dict['category_id']].append('')
            
            #######
            temp_image_list_mode.append(annotations_dict['image_id'])
            out_annotations_list.append(annotations_dict)

        ####### check video

        distinct_image_list_total = list(set(temp_image_list_total))
        distinct_image_list_mode = list(set(temp_image_list_mode))

        image_list_total.extend(distinct_image_list_total)
        image_list_mode.extend(distinct_image_list_mode)

        num_total_annotation += len(annotations_list)
        num_mode_annotation += len(out_annotations_list)
        
        print('video_name :', video_name, '\n',
                '/ previous remove : num_anno :', len(annotations_list), '\n',
                '/ after remove : num_anno :', len(out_annotations_list), '\n',
                '/ previous remove : num_image :', len(distinct_image_list_total), '\n',
                '/ after remove : num_image :', len(distinct_image_list_mode), '\n',)

    ####### check video
    distinct_image_list_total = list(set(image_list_total))
    distinct_image_list_mode = list(set(image_list_mode))

    print('previous remove : num_total_anno :', num_total_annotation, '\n',
            'after remove : num_total_anno :', num_mode_annotation, '\n',
            'previous remove : num_total_image :', len(distinct_image_list_total), '\n',
            'after remove : num_total_image :', len(distinct_image_list_mode), '\n',)

    for class_id_key in check_num_class_dict:
        print(class_id_key, ': num_class :', len(check_num_class_dict[class_id_key]))
        num_sum_classes += len(check_num_class_dict[class_id_key])
    print('num_sum_classes : ', num_sum_classes, '\n',)

    return output_dict


###################################################################################################################
# json (deetas) - load & write
###################################################################################################################
def load_single_json(load_full_path):
    print('****************************************************************************')
    print('annotation path :', load_full_path, '\n')
    
    ### initial output
    output_dict = {}

    ### read
    json_reader = open(load_full_path,'r')
    file_line = json_reader.readline()
    json_data = json.loads(file_line)

    ### initial
    output_dict = json_data

    return output_dict


def find_json_paths(json_root_path):
    print('****************************************************************************')
    print('find json paths in :', json_root_path, '\n')
    ####### initial
    output_path_list = []

    ####### repeat date
    json_name_list = os.listdir(json_root_path)

    for json_name in json_name_list:
        json_full_path = os.path.join(json_root_path, json_name)
        output_path_list.append(json_full_path)

    return output_path_list


def load_multi_json(input_json_paths_list):
    print('*********************************************************************')
    print('load_multi_json', '\n',)

    output_dict = {}

    for json_file_path in input_json_paths_list:
        json_file = open(json_file_path,'r')
        file_line = json_file.readline()
        idx_strip = json_file_path.rfind('/')
        json_file_name = json_file_path[idx_strip+1:]

        try:
            json_data = json.loads(file_line)
            print('load :', json_file_name, 'success')
        except:
            print('load :', json_file_name, ': error !!! error !!! error !!!')
            continue

        ### initial
        output_dict[json_file_name] = json_data

    return output_dict


def write_json(output_file_path, json_data):
    print('****************************************************************************')
    fp = open(output_file_path, 'w')
    json.dump(json_data, fp)


###################################################################################################################
# pickle - load & write
###################################################################################################################
def load_pickles(load_root_path):
    print('****************************************************************************')
    print('pickle path :', load_root_path, '\n')
    output_dict = {}
    
    load_json_train_path = os.path.join(load_root_path, 'train.pickle')
    load_json_val_path = os.path.join(load_root_path, 'val.pickle')
    load_json_test_path = os.path.join(load_root_path, 'test.pickle')
    
    train_list = load_single_pickle(load_json_train_path)
    output_dict['train'] = train_list
    
    val_list = load_single_pickle(load_json_val_path)
    output_dict['val'] = val_list
    
    test_list = load_single_pickle(load_json_test_path)
    output_dict['test'] = test_list
    
    return output_dict


def load_single_pickle(load_full_path):
    print('****************************************************************************')
    print('pickle path :', load_full_path, '\n')
    
    with open(load_full_path, 'rb') as reader_pickle:
        data_list = pickle.load(reader_pickle)

    print('num_data :', len(data_list), '\n')
    return data_list


def write_pickle(input_dict, pickle_root_path):
    print('****************************************************************************')
    print('write_pickle :', '\n',
            'check path :', pickle_root_path, '\n',)
    
    ### define output path
    output_file_name_list = ['train.pickle', 'val.pickle', 'test.pickle']
    train_path = os.path.join(pickle_root_path, output_file_name_list[0])
    val_path = os.path.join(pickle_root_path, output_file_name_list[1])
    test_path = os.path.join(pickle_root_path, output_file_name_list[2])

    train_data = input_dict['train']
    val_data = input_dict['val']
    test_data = input_dict['test']

    check_to_exist_file(5, train_path)

    train_writer = open(train_path, 'wb')
    pickle.dump(train_data, train_writer)
    train_writer.close()

    val_writer = open(val_path, 'wb')
    pickle.dump(val_data, val_writer)
    val_writer.close()

    test_writer = open(test_path, 'wb')
    pickle.dump(test_data, test_writer)
    test_writer.close()


###################################################################################################################
# numpy - load & write
###################################################################################################################
def load_numpy_file(path_dict):
    print('*********************************************************************')
    print('load_numpy_file :', '\n',)

    load_path = path_dict['act_class']
    load_path = os.path.join(load_path, 'added_image_data.npz')

    load_np = np.load(load_path)

    train_bbox = load_np['train_bbox']
    train_bbox_image = load_np['train_bbox_image']
    train_object_class = load_np['train_object_class']
    train_action_class = load_np['train_action_class']

    test_bbox = load_np['test_bbox']
    test_bbox_image = load_np['test_bbox_image']
    test_object_class = load_np['test_object_class']
    test_action_class = load_np['test_action_class']

    print(train_bbox.shape)
    print(train_bbox_image.shape)
    print(train_object_class.shape)
    print(train_action_class.shape)

    print(test_bbox.shape)
    print(test_bbox_image.shape)
    print(test_object_class.shape)
    print(test_action_class.shape)

    return load_np


###################################################################################################################
# hdf5 - write
###################################################################################################################
def write_hdf5(dataset_dict, model_config_dict, path_dict):
    print('*********************************************************************')
    print('write_hdf5 :', '\n',
            'input keys :', dataset_dict.keys(), '\n',)
    
    output_path = path_dict['output']
    print('output_path : ', output_path, '\n',)

    Num_bbox = model_config_dict['Num_bbox']
    image_width = model_config_dict['width']
    image_height = model_config_dict['height']
    image_channel = model_config_dict['channel']

    ####### each dataset
    for idx_write, splited_key in enumerate(dataset_dict):
        bbox_np = dataset_dict[splited_key]['bbox']
        object_class_np = dataset_dict[splited_key]['object_class']
        action_class_np = dataset_dict[splited_key]['action_class']

        object_class_np = object_class_np.reshape(object_class_np.shape[0], 1)
        action_class_np = action_class_np.reshape(action_class_np.shape[0], 1)

        print('---------', splited_key)
        print('bbox : shape :', bbox_np.shape)
        print('object_class : shape :', object_class_np.shape)
        print('action_class : shape :', action_class_np.shape)
        print('\n')

        if not os.path.isfile(output_path) or idx_write == 0:
            with h5py.File(output_path, 'w') as h5:
                num_data = bbox_np.shape[0]

                h5.create_dataset(splited_key + '_bbox',
                    (num_data, Num_bbox, 4),
                    maxshape=(None, Num_bbox, 4),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_object_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_action_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5[splited_key + '_bbox'][:] = bbox_np
                h5[splited_key + '_object_class'][:] = object_class_np
                h5[splited_key + '_action_class'][:] = action_class_np
                
        else:
            with h5py.File(output_path, 'a') as h5:
                num_data = bbox_np.shape[0]

                h5.create_dataset(splited_key + '_bbox',
                    (num_data, Num_bbox, 4),
                    maxshape=(None, Num_bbox, 4),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_object_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_action_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5[splited_key + '_bbox'][:] = bbox_np
                h5[splited_key + '_object_class'][:] = object_class_np
                h5[splited_key + '_action_class'][:] = action_class_np


def write_hdf5_with_image(dataset_dict, model_config_dict, path_dict):
    print('*********************************************************************')
    print('write_hdf5 :', '\n',
            'input keys :', dataset_dict.keys(), '\n',)
    
    output_path = path_dict['output']
    print('output_path : ', output_path, '\n',)

    Num_bbox = model_config_dict['Num_bbox']
    image_width = model_config_dict['width']
    image_height = model_config_dict['height']
    image_channel = model_config_dict['channel']

    ####### each dataset
    for idx_write, splited_key in enumerate(dataset_dict):
        bbox_np = dataset_dict[splited_key]['bbox']
        cropped_image_np = dataset_dict[splited_key]['cropped_image']
        object_class_np = dataset_dict[splited_key]['object_class']
        action_class_np = dataset_dict[splited_key]['action_class']

        object_class_np = object_class_np.reshape(object_class_np.shape[0], 1)
        action_class_np = action_class_np.reshape(action_class_np.shape[0], 1)

        print('---------', splited_key)
        print('bbox : shape :', bbox_np.shape)
        print('cropped_image : shape :', cropped_image_np.shape)
        print('object_class : shape :', object_class_np.shape)
        print('action_class : shape :', action_class_np.shape)
        print('\n')

        if not os.path.isfile(output_path) or idx_write == 0:
            with h5py.File(output_path, 'w') as h5:
                num_data = bbox_np.shape[0]

                h5.create_dataset(splited_key + '_bbox',
                    (num_data, Num_bbox, 4),
                    maxshape=(None, Num_bbox, 4),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_bbox_image',
                    (num_data, Num_bbox, image_width, image_height, image_channel),
                    maxshape=(None, Num_bbox, image_width, image_height, image_channel),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_object_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_action_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5[splited_key + '_bbox'][:] = bbox_np
                h5[splited_key + '_bbox_image'][:] = cropped_image_np
                h5[splited_key + '_object_class'][:] = object_class_np
                h5[splited_key + '_action_class'][:] = action_class_np
                
        else:
            with h5py.File(output_path, 'a') as h5:
                num_data = bbox_np.shape[0]

                h5.create_dataset(splited_key + '_bbox',
                    (num_data, Num_bbox, 4),
                    maxshape=(None, Num_bbox, 4),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_bbox_image',
                    (num_data, Num_bbox, image_width, image_height, image_channel),
                    maxshape=(None, Num_bbox, image_width, image_height, image_channel),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_object_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5.create_dataset(splited_key + '_action_class',
                    (num_data, 1),
                    maxshape=(None, 1),
                    dtype='float32', chunks=True)

                h5[splited_key + '_bbox'][:] = bbox_np
                h5[splited_key + '_bbox_image'][:] = cropped_image_np
                h5[splited_key + '_object_class'][:] = object_class_np
                h5[splited_key + '_action_class'][:] = action_class_np


def write_hdf5_per_step(input_dict, idx_split, idx_iter, splited_name, path_dict):
    output_path = path_dict['output']
    
    file_name = 'cropped_image_train.hdf5'
    output_path = os.path.join(output_path, file_name)

    bbox = input_dict['bbox']
    bbox_image = input_dict['bbox_image']
    object_class = np.expand_dims(input_dict['object_class'], axis=-1)
    action_class = np.expand_dims(input_dict['action_class'], axis=-1)

    print('bbox : ', bbox.shape)
    print('bbox_image : ', bbox_image.shape)
    print('object_class : ', object_class.shape)
    print('action_class : ', action_class.shape)

    num_data = bbox.shape[0]
    if num_data == 0:
        return False
        
    if idx_split == 0 and idx_iter == 0:
        with h5py.File(output_path, 'w') as h5:
            if splited_name == 'train':
                h5.create_dataset('train_bbox', (num_data, 5, 4), maxshape=(None, 5, 4), dtype='float32', chunks=True)
                h5.create_dataset('train_bbox_image', (num_data, 5, 256, 256, 3), maxshape=(None, 5, 256, 256, 3), dtype='float32', chunks=True)
                h5.create_dataset('train_object_class', (num_data, 1), maxshape=(None, 1), dtype='float32', chunks=True)
                h5.create_dataset('train_action_class', (num_data, 1), maxshape=(None, 1), dtype='float32', chunks=True)

                h5['train_bbox'][:] = bbox
                h5['train_bbox_image'][:] = bbox_image
                h5['train_object_class'][:] = object_class
                h5['train_action_class'][:] = action_class

            else:
                print('error!! : splited_name')

    elif idx_iter == 0:
        with h5py.File(output_path, 'a') as h5:
            if splited_name == 'val':
                h5.create_dataset('val_bbox', (num_data, 5, 4), maxshape=(None, 5, 4), dtype='float32', chunks=True)
                h5.create_dataset('val_bbox_image', (num_data, 5, 256, 256, 3), maxshape=(None, 5, 256, 256, 3), dtype='float32', chunks=True)
                h5.create_dataset('val_object_class', (num_data, 1), maxshape=(None, 1), dtype='float32', chunks=True)
                h5.create_dataset('val_action_class', (num_data, 1), maxshape=(None, 1), dtype='float32', chunks=True)

                h5['val_bbox'][:] = bbox
                h5['val_bbox_image'][:] = bbox_image
                h5['val_object_class'][:] = object_class
                h5['val_action_class'][:] = action_class
                
            elif splited_name == 'test':
                h5.create_dataset('test_bbox', (num_data, 5, 4), maxshape=(None, 5, 4), dtype='float32', chunks=True)
                h5.create_dataset('test_bbox_image', (num_data, 5, 256, 256, 3), maxshape=(None, 5, 256, 256, 3), dtype='float32', chunks=True)
                h5.create_dataset('test_object_class', (num_data, 1), maxshape=(None, 1), dtype='float32', chunks=True)
                h5.create_dataset('test_action_class', (num_data, 1), maxshape=(None, 1), dtype='float32', chunks=True)

                h5['test_bbox'][:] = bbox
                h5['test_bbox_image'][:] = bbox_image
                h5['test_object_class'][:] = object_class
                h5['test_action_class'][:] = action_class

            else:
                print('error!! : splited_name')

    else:
        with h5py.File(output_path, 'a') as h5:
            if splited_name == 'train':
                h5["train_bbox"].resize((h5["train_bbox"].shape[0] + num_data), axis = 0)
                h5["train_bbox"][-num_data:] = bbox

                h5["train_bbox_image"].resize((h5["train_bbox_image"].shape[0] + num_data), axis = 0)
                h5["train_bbox_image"][-num_data:] = bbox_image

                h5["train_object_class"].resize((h5["train_object_class"].shape[0] + num_data), axis = 0)
                h5["train_object_class"][-num_data:] = object_class

                h5["train_action_class"].resize((h5["train_action_class"].shape[0] + num_data), axis = 0)
                h5["train_action_class"][-num_data:] = action_class

            elif splited_name == 'val':
                h5["val_bbox"].resize((h5["val_bbox"].shape[0] + num_data), axis = 0)
                h5["val_bbox"][-num_data:] = bbox

                h5["val_bbox_image"].resize((h5["val_bbox_image"].shape[0] + num_data), axis = 0)
                h5["val_bbox_image"][-num_data:] = bbox_image

                h5["val_object_class"].resize((h5["val_object_class"].shape[0] + num_data), axis = 0)
                h5["val_object_class"][-num_data:] = object_class

                h5["val_action_class"].resize((h5["val_action_class"].shape[0] + num_data), axis = 0)
                h5["val_action_class"][-num_data:] = action_class

            elif splited_name == 'test':
                h5["test_bbox"].resize((h5["test_bbox"].shape[0] + num_data), axis = 0)
                h5["test_bbox"][-num_data:] = bbox

                h5["test_bbox_image"].resize((h5["test_bbox_image"].shape[0] + num_data), axis = 0)
                h5["test_bbox_image"][-num_data:] = bbox_image

                h5["test_object_class"].resize((h5["test_object_class"].shape[0] + num_data), axis = 0)
                h5["test_object_class"][-num_data:] = object_class

                h5["test_action_class"].resize((h5["test_action_class"].shape[0] + num_data), axis = 0)
                h5["test_action_class"][-num_data:] = action_class

            else:
                print('error!! : splited_name')

    print('save : complete')


###################################################################################################################
# other
###################################################################################################################
def check_to_exist_file(num_delay, train_path):
    if os.path.isfile(train_path):
        for count in range(num_delay):
            print('already exist file check your model to train : ', num_delay - count)
            time.sleep(1)


###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    pass