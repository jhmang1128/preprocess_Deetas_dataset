# -*- coding: utf-8 -*-
from module import common_process as common_p
from module import class_process as class_p

import argparse
import os

import time

import cv2
import numpy as np

from operator import itemgetter
import random
random.seed(777)

def main():
    ##############################################################################################
    # initial parameter
    ##############################################################################################
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN.')

    parser.add_argument('--load_path', required=True, help="load root path of annotation files")
    parser.add_argument('--load_image_path', required=True, help="load root path of image files")
    parser.add_argument('--save_path', required=True, help='save path of hdf5 file')
    parser.add_argument('--process_mode', required=True, help='select dynamic_action or static_action')
    parser.add_argument('--crop_image_mode', required=True, help="None or select image_padding or image_resize")
    parser.add_argument('--crop_width', required=True, type=int, metavar="256", help='width of image')
    parser.add_argument('--crop_height', required=True, type=int, metavar="256", help='height of image')

    args = parser.parse_args()
    print("load_path : ", args.load_path)
    print("load_image_path : ", args.load_image_path)
    print("save_path : ", args.save_path)
    print("process_mode : ", args.process_mode)
    print("crop_image_mode : ", args.crop_image_mode)
    print("crop_width : ", args.crop_width)
    print("crop_height : ", args.crop_height)

    LOAD_PATH = args.load_path
    LOAD_IMAGE_PATH = args.load_image_path
    SAVE_PATH = args.save_path
    PROPROCESS_MODE = args.process_mode
    CROP_IMAGE_MODE = args.crop_image_mode
    CROP_WIDTH = args.crop_width
    CROP_HEIGHT = args.crop_height

    model_config_dict = {}
    path_dict = {}
    
    ####### image config
    Num_bbox = 5
    num_stride = 2
    
    ####### dict (just deliver to module)
    model_config_dict['crop_image_mode'] = CROP_IMAGE_MODE
    model_config_dict['Num_bbox'] = Num_bbox
    model_config_dict['width'] = CROP_WIDTH
    model_config_dict['height'] = CROP_HEIGHT
    model_config_dict['channel'] = 3

    ##############################################################################################
    # define path and file config
    ##############################################################################################
    print('****************************************************************************')
    print('main', '\n')
    
    ####### path_dict (just deliver to module)
    path_dict['image_root_path'] = LOAD_IMAGE_PATH
    path_dict['output'] = SAVE_PATH
    
    ####### load json
    json_paths_list = common_p.find_json_paths(LOAD_PATH)
    data_dict = common_p.load_multi_json(json_paths_list)

    ####### preprocess
    data_dict = common_p.reset_image_id(data_dict)
    data_dict = common_p.preprocess_annotation(PROPROCESS_MODE, data_dict)
    data_dict = transform_unit_track(data_dict)
    data_dict = split_unit_track(data_dict)
    # data_dict = split_unit_action_class(data_dict, model_config_dict)
    # exit()

    if CROP_IMAGE_MODE == 'None':
        data_dict = sliding_window(num_stride, data_dict, model_config_dict)
        common_p.write_hdf5(data_dict, model_config_dict, path_dict)
    elif CROP_IMAGE_MODE == 'image_resize' or 'image_padding':
        data_image_dict = add_cropped_image(data_dict, model_config_dict, path_dict)
        data_image_dict = sliding_window_with_image(num_stride, data_dict, model_config_dict)
        common_p.write_hdf5_with_image(data_image_dict, model_config_dict, path_dict)
    else:
        print('error : crop_image_mode : typing wrong parameter : please type None or image_resize or image_padding')


###################################################################################################################
# sliding winodw
###################################################################################################################
def sliding_window(num_stride, dataset_dict, model_config_dict):
    print('****************************************************************************')
    print('sliding_window', '\n',
            'input keys :', dataset_dict.keys(), '\n',)
            
    ####### initial
    Num_bbox = model_config_dict['Num_bbox']
    output_dict = {}

    ####### train, validation, test
    for splited_key in dataset_dict:
        output_dict[splited_key] = {}
        output_dict[splited_key]['bbox'] = []
        output_dict[splited_key]['object_class'] = []
        output_dict[splited_key]['action_class'] = []

        count_num_track = 0
        count_num_image_list = []

        tracks_list = dataset_dict[splited_key]

        ####### repeat (per-track)
        for idx_track, single_track_dict in enumerate(tracks_list):
            check_inner_track = False
            annotations_list = single_track_dict['annotations']

            if len(annotations_list) < Num_bbox : continue

            obect_class = single_track_dict['category_id']

            ####### window sliding
            for idx_anno, annotation in enumerate(annotations_list):
                if idx_anno > len(annotations_list) - Num_bbox : break

                ####### stride
                if idx_anno % num_stride == 0 : pass
                else : continue
                
                check_inner_break = False
                batch_image_id = []
                batch_bbox = []
                batch_act_class = []
                
                ####### window
                window = annotations_list[idx_anno : idx_anno + Num_bbox]
                for idx_inner_anno, annotation in enumerate(window):
                        
                    action_class = annotation['Status']
                    image_id = annotation['image_id']
                    bbox = annotation['bbox']
                    
                    if not class_p.check_dynamic_act(action_class) : break
                    if not idx_inner_anno == 0:
                        if image_id > previous_image_id + 5 : break
                    previous_image_id = image_id

                    batch_image_id.append(image_id)
                    batch_bbox.append(bbox)
                    batch_act_class.append(action_class)

                    ####### final in batch
                    if idx_inner_anno == Num_bbox - 1:
                        set_batch = list(set(batch_act_class))
                        if not len(set_batch) == 1 : break
                        check_inner_break = True
                    
                if check_inner_break:
                    check_inner_track = True
                    action_class = class_p.convert_action_class(batch_act_class[0])
                    count_num_image_list.extend(batch_image_id)
                    output_dict[splited_key]['bbox'].append(batch_bbox)
                    output_dict[splited_key]['object_class'].append(obect_class)
                    output_dict[splited_key]['action_class'].append(action_class)

            if check_inner_track:
                count_num_track += 1

        ####### shuffle (each dataset (train, vaidation, test))
        index_list = np.arange(len(output_dict[splited_key]['bbox']))

        bbox_np = np.array(output_dict[splited_key]['bbox'])
        object_class_np = np.array(output_dict[splited_key]['object_class'])
        action_class_np = np.array(output_dict[splited_key]['action_class'])

        random.shuffle(index_list)

        output_dict[splited_key]['bbox'] = bbox_np[index_list]
        output_dict[splited_key]['object_class'] = object_class_np[index_list]
        output_dict[splited_key]['action_class'] = action_class_np[index_list]

        print('after : num_track', count_num_track)
        print('after : num_image', len(list(set(count_num_image_list))), '\n',)
        

    return output_dict


def sliding_window_with_image(num_stride, dataset_dict, model_config_dict):
    print('****************************************************************************')
    print('sliding_window', '\n',
            'input keys :', dataset_dict.keys(), '\n',)
            
    ####### initial
    Num_bbox = model_config_dict['Num_bbox']
    output_dict = {}

    ####### train, validation, test
    for splited_key in dataset_dict:
        output_dict[splited_key] = {}
        output_dict[splited_key]['bbox'] = []
        output_dict[splited_key]['cropped_image'] = []
        output_dict[splited_key]['object_class'] = []
        output_dict[splited_key]['action_class'] = []

        count_num_track = 0
        count_num_image_list = []

        tracks_list = dataset_dict[splited_key]

        ####### repeat (per-track)
        for idx_track, single_track_dict in enumerate(tracks_list):
            check_inner_track = False
            annotations_list = single_track_dict['annotations']

            if len(annotations_list) < Num_bbox : continue

            obect_class = single_track_dict['category_id']

            ####### window sliding
            for idx_anno, annotation in enumerate(annotations_list):
                ####### stride
                if idx_anno % num_stride == 0 : pass
                else : continue

                if idx_anno > len(annotations_list) - Num_bbox : break
                
                check_inner_break = False
                batch_image_id = []
                batch_bbox = []
                batch_image = []
                batch_act_class = []
                
                ####### window
                window = annotations_list[idx_anno : idx_anno + Num_bbox]
                for idx_inner_anno, annotation in enumerate(window):
                    action_class = annotation['Status']
                    image_id = annotation['image_id']
                    bbox = annotation['bbox']
                    
                    if not class_p.check_dynamic_act(action_class) : break
                    if not idx_inner_anno == 0:
                        if image_id > previous_image_id + 5 : break
                    if not 'image_in_bbox' in annotation : break

                    cropped_image = annotation['image_in_bbox']
                    previous_image_id = image_id

                    batch_image_id.append(image_id)
                    batch_bbox.append(bbox)
                    batch_image.append(cropped_image)
                    batch_act_class.append(action_class)

                    ####### final in batch
                    if idx_inner_anno == Num_bbox - 1:
                        set_batch = list(set(batch_act_class))
                        if not len(set_batch) == 1 : break
                        check_inner_break = True
                    
                if check_inner_break:
                    check_inner_track = True
                    action_class = class_p.convert_action_class(batch_act_class[0])
                    count_num_image_list.extend(batch_image_id)
                    output_dict[splited_key]['bbox'].append(batch_bbox)
                    output_dict[splited_key]['cropped_image'].append(batch_image)
                    output_dict[splited_key]['object_class'].append(obect_class)
                    output_dict[splited_key]['action_class'].append(action_class)

            if check_inner_track:
                count_num_track += 1

        ####### shuffle (each dataset (train, vaidation, test))
        index_list = np.arange(len(output_dict[splited_key]['bbox']))

        bbox_np = np.array(output_dict[splited_key]['bbox'])
        cropped_image_np = np.array(output_dict[splited_key]['cropped_image'])
        object_class_np = np.array(output_dict[splited_key]['object_class'])
        action_class_np = np.array(output_dict[splited_key]['action_class'])

        random.shuffle(index_list)

        output_dict[splited_key]['bbox'] = bbox_np[index_list]
        output_dict[splited_key]['cropped_image'] = cropped_image_np[index_list]
        output_dict[splited_key]['object_class'] = object_class_np[index_list]
        output_dict[splited_key]['action_class'] = action_class_np[index_list]

        print('after : num_track', count_num_track)
        print('after : num_image', len(list(set(count_num_image_list))), '\n',)

    return output_dict


###################################################################################################################
# add image
###################################################################################################################
def add_cropped_image(dataset_dict, model_config_dict, path_dict):
    print('****************************************************************************')
    print('add_cropped_image', '\n',
            'input keys :', dataset_dict.keys(), '\n',)
            
    ####### initial
    image_root_path = path_dict['image_root_path']

    ####### train, validation, test
    for splited_key in dataset_dict:
        tracks_list = dataset_dict[splited_key]

        ####### repeat (per-track)
        for idx_track, single_track_dict in enumerate(tracks_list):
            if idx_track % 100 == 0:
                print(splited_key, ': processing... : idx_track :', idx_track, '~')
            # if idx_track == 10 : # test
            #     break
            annotations_list = single_track_dict['annotations']

            ####### repeat (per-batch)
            for annotation in annotations_list:
                bbox_value = annotation['bbox']
                image_file_name = annotation['image_file_name']
                image_path = os.path.join(image_root_path, image_file_name)
                if not os.path.isfile(image_path):
                    print('image_path :', image_path, ': not exist')
                    break
                image_in_bbox = crop_image_in_bbox(bbox_value, image_path, model_config_dict)
                annotation['image_in_bbox'] = image_in_bbox

                # if not annotation.get('image_in_bbox'):
                #     annotation['image_in_bbox'] = []
                # annotation['image_in_bbox'].append()

    return dataset_dict


def crop_image_in_bbox(bounding_box, image_path, model_config_dict):
    crop_image_mode = model_config_dict['crop_image_mode']
    image_width = model_config_dict['width']
    image_height = model_config_dict['height']

    cood_X = int(bounding_box[0])
    cood_Y = int(bounding_box[1])
    width = int(bounding_box[2])
    height = int(bounding_box[3])

    # image_pli = PIL.Image.open(image_path)
    image_np = cv2.imread(image_path)
    cropped_image = image_np[cood_Y : cood_Y + height, cood_X : cood_X + width]

    if crop_image_mode == 'image_padding':
        ####### padding
        if height > width:
            padding_image = np.zeros(shape=(height, height, 3))
            padding_image[:, int((height-width)/2) : int((height-width)/2) + width] = cropped_image
        else:
            padding_image = np.zeros(shape=(width, width, 3))
            padding_image[int((width-height)/2) : int((width-height)/2) + height, :] = cropped_image

        resized_image = cv2.resize(padding_image, dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC)

    elif crop_image_mode == 'image_resize':
        resized_image = cv2.resize(cropped_image, dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC)

    cropped_image = resized_image

    return cropped_image


###################################################################################################################
# split and merge (dict, .json)
###################################################################################################################
def split_unit_action_class(dataset_dict, model_config_dict):
    print('*********************************************************************')
    print('split_unit_action_class :', '\n',
            'input keys :', dataset_dict.keys(), '\n',)

    ####### initial
    Num_bbox = model_config_dict['Num_bbox']
    output_dict = {}
    output_dict['train'] = []
    output_dict['val'] = []
    output_dict['test'] = []

    count_sum_Going = 0
    count_sum_Coming = 0
    count_sum_Crossing = 0
    count_sum_Stopping = 0

    ####### train, validation, test
    for splited_key in dataset_dict:
        tracks_list = dataset_dict[splited_key]

        count_Going = 0
        count_Coming = 0
        count_Crossing = 0
        count_Stopping = 0

        ####### repeat (per-track)
        for idx_track, single_track_dict in enumerate(tracks_list):
            obejct_id = single_track_dict['id']
            category_id = single_track_dict['category_id']
            annotations_list = single_track_dict['annotations']

            status_in_track_list = []
            previous_status_class = False
            for idx_anno, annotation in enumerate(annotations_list):
                status_class = annotation['Status']

                if status_class == previous_status_class or idx_anno == 0:
                    status_in_track_list.append(annotation)
                else:
                    if len(status_in_track_list) < Num_bbox : pass
                    else :
                        output_dict[splited_key].append(status_in_track_list)
                        for annotation in status_in_track_list:
                            check_status_class = annotation['Status']
                            if check_status_class == 'Going': count_Going += 1
                            elif check_status_class == 'Coming': count_Coming += 1
                            elif check_status_class == 'Crossing': count_Crossing += 1
                            elif check_status_class == 'Stopping': count_Stopping += 1
                            else:
                                print('error : status class')
                            break # one

                    status_in_track_list = []
                
                previous_status_class = status_class

        print('count_status :', splited_key, ': Going :', count_Going)
        print('count_status :', splited_key, ': Coming :', count_Coming)
        print('count_status :', splited_key, ': Crossing :', count_Crossing)
        print('count_status :', splited_key, ': Stopping :', count_Stopping)
        print('\n')

        count_sum_Going += count_Going
        count_sum_Coming += count_Coming
        count_sum_Crossing += count_Crossing
        count_sum_Stopping += count_Stopping

    print('count_status : sum : Going :', count_sum_Going)
    print('count_status : sum : Coming :', count_sum_Coming)
    print('count_status : sum : Crossing :', count_sum_Crossing)
    print('count_status : sum : Stopping :', count_sum_Stopping)
    print('\n')

    print('after : num_track : train :', len(output_dict['train']))
    print('after : num_track : val :', len(output_dict['val']))
    print('after : num_track : test :', len(output_dict['test']))
    print('after : num_track : sum :', len(output_dict['train']) + len(output_dict['val']) + len(output_dict['test']))
    print('\n')
        
    return output_dict


def split_unit_track(videos_track_dict):
    print('*********************************************************************')
    print('split_unit_video :', '\n',
            'input keys :', videos_track_dict.keys(), '\n',)

    ####### initial
    output_dict = {}
    output_keys_list = ['train', 'val', 'test']
    for splited_key in output_keys_list:
        output_dict[splited_key] = []
    temp_videos_list = []
    
    ####### insert video to list
    for idx_video, video_name in enumerate(videos_track_dict):
        temp_videos_list.append(videos_track_dict[video_name])
        tracks_list = videos_track_dict[video_name]
        
        ####### split
        random.shuffle(tracks_list)
        num_videos = len(tracks_list)
        idx_01 = int(num_videos*0.8)
        idx_02 = int(num_videos*0.9)

        if not idx_video == 0:
            train_tracks_list.extend(tracks_list[:idx_01])
            val_tracks_list.extend(tracks_list[idx_01:idx_02])
            test_tracks_list.extend(tracks_list[idx_02:])

        else:
            train_tracks_list = tracks_list[:idx_01]
            val_tracks_list = tracks_list[idx_01:idx_02]
            test_tracks_list = tracks_list[idx_02:]

    dataset_list = [train_tracks_list, val_tracks_list, test_tracks_list]

    for splited_key, splited_dataset_list in zip(output_keys_list, dataset_list):
        output_dict[splited_key] = splited_dataset_list

    print('output keys :', output_dict.keys(), '\n',)

    print('train : tracks_num :', len(output_dict['train']), '\n',
            'val : tracks_num :', len(output_dict['val']), '\n',
            'test : tracks_num :', len(output_dict['test']), '\n',)

    return output_dict


###################################################################################################################
# preprocess from dict(.json data format) to list(.pickle data format) / (action class, bounding box, object class, image_id)
###################################################################################################################
def transform_unit_track(dataset_dict):
    print('****************************************************************************')
    print('transform_unit_track', '\n',
            'input keys :', dataset_dict.keys(), '\n',)
    
    ####### initial
    output_dict = {}
    sum_num_track = 0

    ####### repeat for json_file
    for json_file_name in dataset_dict:
        
        ####### initial
        images_dict = {}
        output_dict[json_file_name] = []
        video_dict = dataset_dict[json_file_name]
        
        ####### parent_attribute
        categories_list = video_dict['categories']
        images_list = video_dict['images']

        
        for image_dict in images_list:
            images_dict[image_dict['id']] = image_dict['file_name']
        
        annotations_array = video_dict['annotations']

        ####### repeat to read attributes in annotations
        temp_dict = {}
        for annotations_dict in annotations_array:
            if annotations_dict['attributes'].get('track_id'):
                temp_anno_dict = {}
                image_id = annotations_dict['image_id']
                image_file_name = images_dict[image_id]
                segmentation = annotations_dict['segmentation']
                bbox = annotations_dict['bbox']
                track_id = annotations_dict['attributes']['track_id']
                status = annotations_dict['attributes']['Status']
                category_id = annotations_dict['category_id']

                ####### track
                if temp_dict.get(track_id):
                    temp_dict[track_id]['annotations'].append({'image_id' : image_id,
                                                                'image_file_name' : image_file_name,
                                                                'bbox' : bbox,
                                                                'segmentation' : segmentation,
                                                                'Status' : status})
                else:
                    temp_dict[track_id] = {'id' : track_id,
                                            'category_id' : category_id,
                                            'annotations' : [{'image_id' : image_id,
                                                                'image_file_name' : image_file_name,
                                                                'bbox' : bbox,
                                                                'segmentation' : segmentation,
                                                                'Status' : status}]}

        for track_id in temp_dict:
            temp_dict[track_id]['annotations'].sort(key = itemgetter('image_id'), reverse=False)
            output_dict[json_file_name].append(temp_dict[track_id])

        print(json_file_name, ': num_track :', len(output_dict[json_file_name]))
        sum_num_track += len(output_dict[json_file_name])

    print('\n')
    print(' sum of nums_track :', sum_num_track, '\n',)

    return output_dict

###################################################################################################################
# other
###################################################################################################################
def check_to_exist_file(num_delay, train_path):
    if os.path.isfile(train_path):
        for count in range(num_delay):
            print('already exist file check your model to train : ', 10 - count)
            time.sleep(1)


###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    main()