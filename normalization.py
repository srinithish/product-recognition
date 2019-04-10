# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:08:28 2019

@author: Amit
"""

def normalize_coordinates(image_dict,object_dict,grid_row_no,grid_col_no,no_of_row_grids,no_of_col_grids):
    '''
    1. image_dict = {path,width,height,depth} 
    \\ depth is no of channels here
    
    2. object_dict = {class,xmin,ymin,xmax,ymax}
    \\ (xmin,ymin) gives the left-most point of the bounding box
    
    3. grid_row_no = (int) grid row number where the center of falls (indexing from 0)
    
    4. grid_col_no = (int) grid column number where the center falls (indexing from 0)
    
    5. no_of_row_grids = (int) Total number of row grids 
    
    6. no_of_ygrids = (int) Total number of column grids 
    '''
    cell_height = image_dict["height"] / no_of_row_grids
    cell_width = image_dict["width"] / no_of_col_grids
    
    grid_cell_topleft = {}
    grid_cell_topleft["x"] = grid_row_no*cell_height
    grid_cell_topleft["y"] = grid_col_no*cell_width
    
    bound_box_center = {}
    bound_box_center["x"] = (object_dict["xmin"] + object_dict["xmax"])/2
    bound_box_center["y"] = (object_dict["ymin"] + object_dict["ymax"])/2
                            
    normalized_coord = {}
    normalized_coord["x"] = (bound_box_center["x"] - grid_cell_topleft["x"])/cell_width 
    normalized_coord["y"] = (bound_box_center["y"] - grid_cell_topleft["y"])/cell_height

    return normalized_coord


def normalize_box_dimension(image_dict,object_dict):
    '''
    1. image_dict = {path,width,height,depth} 
    \\ depth is no of channels here
    
    2. object_dict = {class,xmin,ymin,xmax,ymax}
    \\ (xmin,ymin) gives the left-most point of the bounding box
    '''
    normalized_dimensions = {}
    # box_width/image_width
    normalized_dimensions["width"] = (object_dict["xmax"]-object_dict["xmin"])/image_dict["width"]
    normalized_dimensions["height"] = (object_dict["ymax"]-object_dict["ymin"])/image_dict["height"]
    
    return normalized_dimensions
    
    
    
    

    
    
    
    