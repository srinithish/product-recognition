# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:22:15 2019

@author: Amit
"""

def denormalize_coordinates(image_dict,normalized_coords,grid_row_no,grid_col_no,total_grid_rows,total_grid_cols):
    '''
    b.b => bounding box
    
    The output of feedforward would be of the dimensions: 
        # of gridrows x # of gridcols x v
        v = prob. of obj + (4 b.b params) + # of classes
        
    This function returns the coordinates for the center of a b.b
    
    The inputs for this function are:
        1. image_dict => represents the size of the image
        2. normalized_coords {x,y} => normalized coordinates between (0,1) 
        3. grid_row_no => The row to which this particular grid cell belongs to
        4. grid_col_no => The col to which this particular grid cell belongs to
        5. total_grid_rows => Total grid cells in a row
        6. total_grid_cols => Total grid cells in a col
    '''
    cell_height = image_dict["height"] / total_grid_rows
    cell_width = image_dict["width"] / total_grid_cols
    
    grid_cell_topleft = {}
    grid_cell_topleft["y"] = grid_row_no*cell_height
    grid_cell_topleft["x"] = grid_col_no*cell_width
    
    denormalized_coord = {}
    denormalized_coord["x"] = int(grid_cell_topleft["x"] + (cell_width*normalized_coords["x"]))
    denormalized_coord["y"] = int(grid_cell_topleft["y"] + (cell_height*normalized_coords["y"]))
    
    return denormalized_coord


def denormalize_box_dimension(image_dict,normalized_dimensions):
    '''
    This function returns the width and height of the b.b
    
    The inputs for this function are:
        1. image_dict => represents the size of the image
        2. normalized_box_dimensions {width,height} => normalized box dimensions between (0,1) 
    '''
    denormalized_dimensions = {}
    
    # box_width/image_width
    denormalized_dimensions["width"] = int(normalized_dimensions["width"]*image_dict["width"])
    denormalized_dimensions["height"] = int(normalized_dimensions["height"]*image_dict["height"])
    
    return denormalized_dimensions
    
    
def getDenormalizedBoxParams(image_dict,normalized_dimensions,normalized_coords,grid_row_no,grid_col_no,total_grid_rows,total_grid_cols):
    box_dimensions = denormalize_box_dimension(image_dict,normalized_dimensions)
    coordinates = denormalize_coordinates(image_dict,normalized_coords,grid_row_no,grid_col_no,total_grid_rows,total_grid_cols)
    
    box_dict = {"bx":coordinates["x"],
                "by":coordinates["y"],
                "bh":box_dimensions["height"],
                "bw":box_dimensions["width"]
                }
    
    return box_dict
