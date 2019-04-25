
# coding: utf-8



def _get_center(bb_x_min, bb_y_min, bb_x_max, bb_y_max):
    x_bb_center = (bb_x_min + bb_x_max) / 2
    y_bb_center = (bb_y_min + bb_y_max) / 2
    
    box_center = (x_bb_center, y_bb_center)
    
    return box_center




def _get_grid_position(image_width, image_height, x_num_grids, y_num_grids, box_center_tuple):
    width_cell = image_width / x_num_grids
    height_cell = image_height / y_num_grids
    
    grid_col_num = box_center_tuple[0] // width_cell
    grid_row_num = box_center_tuple[1] // height_cell
    
    return int(grid_row_num), int(grid_col_num)



#get_grid_position(10,10,10,10, (0.5,0.5))




def assign_grid(image_dict, object_dict, x_num_grids, y_num_grids):
    
    image_class = object_dict['name']
    bb_x_min = object_dict['xmin']
    bb_y_min = object_dict['ymin']
    bb_x_max = object_dict['xmax']
    bb_y_max = object_dict['ymax']

    
    image_path = image_dict['path']
    image_width = image_dict['width']
    image_height = image_dict['height']
    image_depth = image_dict['depth']
    
    box_center_tuple = _get_center(bb_x_min, bb_y_min, bb_x_max, bb_y_max)
#    print(box_center_tuple)
    
    return _get_grid_position(image_width, image_height, x_num_grids, y_num_grids, box_center_tuple)    




if __name__ == '__main__':

    imageDict,objectList = parseXMLtoDict("C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Git Repo/product-recognition/twoObjectsCorrect.xml")
    gridRow,gridCol = assign_grid(imageDict, objectList[0], 3,3)


