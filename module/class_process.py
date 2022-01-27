# -*- coding: utf-8 -*-

'''
    ####### dynamic
    (Going, Coming, Crossing, Stopping)

    [{"id": 1, "name": "Person", "supercategory": ""},
    {"id": 2, "name": "Animal", "supercategory": ""},
    {"id": 3, "name": "Vehicle", "supercategory": ""},
    {"id": 4, "name": "Wheeled Object", "supercategory": ""},
    {"id": 5, "name": "Movable Object", "supercategory": ""},

    ####### static
    (Avoiding, Opening, Closing, Moving, On, Off)

    {"id": 6, "name": "Fixed Object", "supercategory": ""},             # X
    {"id": 7, "name": "Obstruction", "supercategory": ""},              # Avoiding

    {"id": 8, "name": "Automatic Door", "supercategory": ""},           # Opening, Closing
    {"id": 9, "name": "Automatic Revolving Door", "supercategory": ""}, # Moving, Stopping
    {"id": 10, "name": "Sliding Door", "supercategory": ""},            # Opening, Closing
    {"id": 11, "name": "Hinger Door", "supercategory": ""},             # Opening, Closing
    {"id": 12, "name": "Manual Revolving Door", "supercategory": ""},   # Moving, Stopping
    {"id": 13, "name": "Escalator", "supercategory": ""},               # Moving, Stopping
    {"id": 14, "name": "Elevator", "supercategory": ""},                # Opening, Closing

    {"id": 15, "name": "Address", "supercategory": ""},                 # X
    {"id": 16, "name": "Sign", "supercategory": ""},                    # X
    {"id": 17, "name": "Screen", "supercategory": ""},                  # X

    {"id": 18, "name": "Up", "supercategory": ""},                      # On, Off
    {"id": 19, "name": "Down", "supercategory": ""},                    # On, Off
    {"id": 20, "name": "Open", "supercategory": ""},                    # On, Off
    {"id": 21, "name": "Close", "supercategory": ""},                   # On, Off
    {"id": 22, "name": "Floor Button", "supercategory": ""},            # On, Off
    {"id": 23, "name": "Emergency Button", "supercategory": ""},        # On, Off
    {"id": 24, "name": "Handle", "supercategory": ""},                  # X
    {"id": 25, "name": "Bell", "supercategory": ""}]                    # X
'''

###################################################################################################################
# generate_category_ids
###################################################################################################################
def generate_static_except_Button():
    class_list = [
                    # {"id": 1, "name": "Person", "supercategory": ""},
                    # {"id": 2, "name": "Animal", "supercategory": ""},
                    # {"id": 3, "name": "Vehicle", "supercategory": ""},
                    # {"id": 4, "name": "Wheeled Object", "supercategory": ""},
                    # {"id": 5, "name": "Movable Object", "supercategory": ""},

                    {"id": 1, "name": "Fixed Object", "supercategory": ""},
                    {"id": 2, "name": "Obstruction", "supercategory": ""},

                    {"id": 3, "name": "Automatic Door", "supercategory": ""},
                    {"id": 4, "name": "Automatic Revolving Door", "supercategory": ""},
                    {"id": 5, "name": "Sliding Door", "supercategory": ""},
                    {"id": 6, "name": "Hinger Door", "supercategory": ""},
                    {"id": 7, "name": "Manual Revolving Door", "supercategory": ""},
                    {"id": 8, "name": "Escalator", "supercategory": ""},
                    {"id": 9, "name": "Elevator", "supercategory": ""},

                    {"id": 10, "name": "Address", "supercategory": ""},
                    {"id": 11, "name": "Sign", "supercategory": ""},
                    {"id": 12, "name": "Screen", "supercategory": ""},

                    # {"id": 13, "name": "Up", "supercategory": ""},
                    # {"id": 14, "name": "Down", "supercategory": ""},
                    # {"id": 15, "name": "Open", "supercategory": ""},
                    # {"id": 16, "name": "Close", "supercategory": ""},
                    # {"id": 22, "name": "Floor Button", "supercategory": ""},
                    # {"id": 23, "name": "Emergency Button", "supercategory": ""},

                    {"id": 13, "name": "Handle", "supercategory": ""}
                    # {"id": 14, "name": "Bell", "supercategory": ""}
                ]
    
    return class_list


def generate_dynamic_action():
    class_list = [
                    {"id": 1, "name": "Person", "supercategory": ""},
                    {"id": 2, "name": "Animal", "supercategory": ""},
                    {"id": 3, "name": "Vehicle", "supercategory": ""},
                    {"id": 4, "name": "Wheeled Object", "supercategory": ""},
                    {"id": 5, "name": "Movable Object", "supercategory": ""}
                ]
    
    return class_list


def generate_static_action():
    class_list = [
                    # {"id": 1, "name": "Fixed Object", "supercategory": ""},
                    {"id": 1, "name": "Obstruction (Avoiding)", "supercategory": ""},

                    {"id": 2, "name": "Automatic Door (Opening)", "supercategory": ""},
                    {"id": 3, "name": "Automatic Door (Closing)", "supercategory": ""},
                    {"id": 4, "name": "Automatic Revolving Door (Moving)", "supercategory": ""},
                    {"id": 5, "name": "Automatic Revolving Door (Stopping)", "supercategory": ""},
                    {"id": 6, "name": "Sliding Door (Opening)", "supercategory": ""},
                    {"id": 7, "name": "Sliding Door (Closing)", "supercategory": ""},
                    {"id": 8, "name": "Hinger Door (Opening)", "supercategory": ""},
                    {"id": 9, "name": "Hinger Door (Closing)", "supercategory": ""},
                    {"id": 10, "name": "Manual Revolving Door (Moving)", "supercategory": ""},
                    {"id": 11, "name": "Manual Revolving Door (Stopping)", "supercategory": ""},
                    {"id": 12, "name": "Escalator (Moving)", "supercategory": ""},
                    {"id": 13, "name": "Escalator (Stopping)", "supercategory": ""},
                    {"id": 14, "name": "Elevator (Opening)", "supercategory": ""},
                    {"id": 15, "name": "Elevator (Closing)", "supercategory": ""}

                    # {"id": 17, "name": "Address", "supercategory": ""},
                    # {"id": 18, "name": "Sign", "supercategory": ""},
                    # {"id": 19, "name": "Screen", "supercategory": ""},

                    # {"id": 20, "name": "Up (On)", "supercategory": ""},
                    # {"id": 21, "name": "Up (Off)", "supercategory": ""},
                    # {"id": 22, "name": "Down (On)", "supercategory": ""},
                    # {"id": 23, "name": "Down (Off)", "supercategory": ""},
                    # {"id": 24, "name": "Open (On)", "supercategory": ""},
                    # {"id": 25, "name": "Open (Off)", "supercategory": ""},
                    # {"id": 26, "name": "Close (On)", "supercategory": ""},
                    # {"id": 27, "name": "Close (Off)", "supercategory": ""},
                    # {"id": 28, "name": "Floor Button (On)", "supercategory": ""},
                    # {"id": 29, "name": "Floor Button (Off)", "supercategory": ""},
                    # {"id": 30, "name": "Emergency Button (On)", "supercategory": ""},
                    # {"id": 31, "name": "Emergency Button (Off)", "supercategory": ""},

                    # {"id": 20, "name": "Handle", "supercategory": ""},
                    # {"id": 21, "name": "Bell", "supercategory": ""}
                ]
    
    return class_list


###################################################################################################################
# convert class_id of anntation 
###################################################################################################################
def converted_into_static_action(annotations_dict):
    num_dynamic = 5
    class_id = annotations_dict['category_id'] - 5
    status_id = annotations_dict['attributes']['Status']

    if class_id == 1: return False
    elif class_id == 2: class_id = 1
    ### Automatic Door
    elif class_id == 3:
        if status_id == 'Opening': class_id = 2
        elif status_id == 'Closing': class_id = 3
        else : return False
        # else : print('error : class_id and status :' , class_id, status_id)
    ### Automatic Revolving Door
    elif class_id == 4: 
        if status_id == 'Moving': class_id = 4
        elif status_id == 'Stopping': class_id = 5
        else : return False
    ### Sliding Door
    elif class_id == 5: 
        if status_id == 'Opening': class_id = 6
        elif status_id == 'Closing': class_id = 7
        else : return False
    ### Hinger Door
    elif class_id == 6: 
        if status_id == 'Opening': class_id = 8
        elif status_id == 'Closing': class_id = 9
        else : return False
    ### Manual Revolving Door
    elif class_id == 7: 
        if status_id == 'Moving': class_id = 10
        elif status_id == 'Stopping': class_id = 11
        else : return False
        # else : print('error : class_id and status :' , class_id, status_id)
    ### Escalator
    elif class_id == 8: 
        if status_id == 'Moving': class_id = 12
        elif status_id == 'Stopping': class_id = 13
        else : return False
    ### Elevator
    elif class_id == 9: 
        if status_id == 'Opening': class_id = 14
        elif status_id == 'Closing': class_id = 15
        else : return False

    ### Address
    elif class_id == 10 : return False
    ### Sign
    elif class_id == 11 : return False
    ### Screen
    elif class_id == 12 : return False

    ### Up
    elif class_id == 13: return False
        # if status_id == 'On': class_id = 20
        # elif status_id == 'Off': class_id = 21
        # else : return False
    ### Down
    elif class_id == 14: return False
        # if status_id == 'On': class_id = 22
        # elif status_id == 'Off': class_id = 23
        # else : return False
    ### Open
    elif class_id == 15: return False
        # if status_id == 'On': class_id = 24
        # elif status_id == 'Off': class_id = 25
        # else : return False
    ### Close
    elif class_id == 16: return False
        # if status_id == 'On': class_id = 26
        # elif status_id == 'Off': class_id = 27
        # else : return False
    ### Floor Button
    elif class_id == 17: return False
        # if status_id == 'On': class_id = 28
        # elif status_id == 'Off': class_id = 29
        # else : return False
    ### Emergency Button
    elif class_id == 18: return False
        # if status_id == 'On': class_id = 30
        # elif status_id == 'Off': class_id = 31
        # else : return False

    ### Handle
    elif class_id == 19 : return False
    ### Bell
    elif class_id == 20 : return False

    annotations_dict['category_id'] = class_id
    
    return annotations_dict


###################################################################################################################
# check action class
###################################################################################################################
def count_object_static(action_class, action_class_dict):
    if action_class == 1 : action_class_dict['Fixed Object'] += 1
    elif action_class == 2 : action_class_dict['Obstruction'] += 1
    elif action_class == 3 : action_class_dict['Automatic Door'] += 1
    elif action_class == 4 : action_class_dict['Automatic Revolving Door'] += 1
    elif action_class == 5 : action_class_dict['Sliding Door'] += 1
    elif action_class == 6 : action_class_dict['Hinger Door'] += 1
    elif action_class == 7 : action_class_dict['Manual Revolving Door'] += 1
    elif action_class == 8 : action_class_dict['Escalator'] += 1
    elif action_class == 9 : action_class_dict['Elevator'] += 1
    elif action_class == 10 : action_class_dict['Address'] += 1
    elif action_class == 11 : action_class_dict['Sign'] += 1
    elif action_class == 12 : action_class_dict['Screen'] += 1
    elif action_class == 13 : action_class_dict['Handle'] += 1
    elif action_class == 14 : action_class_dict['Bell'] += 1
    else:
        print('error : act_class')
        exit()

    return action_class_dict


def count_action_class(action_class, action_class_dict):
    if action_class == 'Going' : action_class_dict['Going'] += 1
    elif action_class == 'Coming' : action_class_dict['Coming'] += 1
    elif action_class == 'Crossing' : action_class_dict['Crossing'] += 1
    elif action_class == 'Stopping' : action_class_dict['Stopping'] += 1
    elif action_class == 'Moving' : action_class_dict['Moving'] += 1
    elif action_class == 'Avoiding' : action_class_dict['Avoiding'] += 1
    elif action_class == 'Opening' : action_class_dict['Opening'] += 1
    elif action_class == 'Closing' : action_class_dict['Closing'] += 1
    elif action_class == 'On' : action_class_dict['On'] += 1
    elif action_class == 'Off' : action_class_dict['Off'] += 1
    else:
        print('error : act_class')
        exit()

    return action_class_dict


def check_bool_dynamic_action(action_class):
    '''
        True : use
        False : skip
    '''
    if action_class == 'Going' : return True
    elif action_class == 'Coming' : return True
    elif action_class == 'Crossing' : return True
    elif action_class == 'Stopping' : return True
    else : return False
    # elif action_class == 'Moving' : return False
    # elif action_class == 'Avoiding' : return False
    # elif action_class == 'Opening' : return False
    # elif action_class == 'Closing' : return False
    # elif action_class == 'On' : return False
    # elif action_class == 'Off' : return False


def check_dynamic_act(action_class):
    if action_class == 'Going': return True
    elif action_class == 'Coming': return True
    elif action_class == 'Crossing': return True
    elif action_class == 'Stopping': return True
    else : return False


def convert_action_class (action_class):
    if action_class == 'Going' : output = 0
    elif action_class == 'Coming' : output = 1
    elif action_class == 'Crossing' : output = 2
    elif action_class == 'Stopping' : output = 3
    elif action_class == 'Moving' : output = 4
    elif action_class == 'Stoping' : output = 5
    elif action_class == 'Avoiding' : output = 6
    elif action_class == 'Opening' : output = 7
    elif action_class == 'Closing' : output = 8
    elif action_class == 'On' : output = 9
    elif action_class == 'Off' : output = 10
    else : 
        print('error : act_class')
        exit()

    return output


###################################################################################################################
# end
###################################################################################################################
if __name__=='__main__':
    pass
