from pdf2image import convert_from_path, convert_from_bytes
from IPython.display import display, Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import s3fs
import boto3
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from google.colab.patches import cv2_imshow
import cv2
import re

def convert_pdf_to_jpegs(pdf_file_name):

    if "pdf_sheets" not in os.listdir("/content"):
        print('Making directory /content/pdf_sheets')
        os.mkdir("/content/pdf_sheets")
    else:
        print('Removing and making directory /content/pdf_sheets')
        os.system("rm -r /content/pdf_sheets")
        os.mkdir("/content/pdf_sheets")

    print('Converting PDF sheets to JPEGs')
    images = convert_from_bytes(open(pdf_file_name,'rb').read(), 
                                fmt='jpeg',
                                size = (8800, 6800), 
                                #first_page=0, last_page=5,
                                output_folder="/content/pdf_sheets/",
                                output_file='pdf_sheet',
                                paths_only=True)

def convert_pdf_to_image(fn, width, height):
    """
    Take in a file name and convert the PDF into a 
    list of grayscale images
    """
    images = convert_from_bytes(open(fn,'rb').read(), size = (width, height))
    
    for i in range(len(images)):
        images_i = images[i].convert('L')
        images[i] = images_i

    return images

def convert_image_to_array(image_list):
    """
        Convert PIL Image to list of numpy arrays
    """

    array_list = []

    for i in range(len(image_list)):
        array_i = np.array(image_list[i])
        array_list.append(array_i)

    return array_list

def get_row_starts(img_raw, cutoff):
    """
        Identify the darker pixel rows, which indicate a line
        Return the start of the rows
    """

    dark_rows = np.where(np.mean(img_raw, 1) <= cutoff)[0]
    dark_row_starts = np.ediff1d(dark_rows, to_begin=dark_rows[0]-1) - 1
    dark_row_starts = dark_row_starts>0
    return dark_rows[dark_row_starts]

def crop_img_rows(img_raw, row_starts):

    """
        Take in a numpy array and crop the array based on the 
        supplied array of indicies that determine the start
        of a new property row.
    """
    assert isinstance(img_raw, (np.ndarray))
    start_idx = 0
    stop_idx = 1
    
    # calculate the median row height, add this value to the last
    # value in row_starts, and append to row_start in order
    # to crop the last row in the sheet
    med_row_height = np.median((np.ediff1d(row_starts)))
    final_crop_end = med_row_height + row_starts[-1]
    
    row_starts=np.append(row_starts, final_crop_end.astype(int))
        
    img_crop_l = []
    
    for i in range(len(row_starts)-1):
        crop_start = row_starts[start_idx]
        # Determine which pixel row is completeley blank, which 
        # represents the end of the horizontal line.
        if crop_start+25 <= img_raw.shape[0]:
            for h in range(crop_start, crop_start+25):
                if np.mean(img_raw[h,:]) >= 250:
                    crop_start = h
        crop_end = row_starts[stop_idx] - 1
        img_crop_i = img_raw[crop_start:crop_end,:]
        
        # Ensure that the horizontal bar has been cropped out
        #assert np.mean(img_crop_i,1)[0] == 255.0

        img_crop_l.append(img_crop_i)
        start_idx += 1
        stop_idx += 1
            
    return img_crop_l

def identify_header_row(row_starts):
    header_idx = 0
    diff = np.Inf
    for i in range(1, len(row_starts)):
        diff_i = row_starts[i]-row_starts[i-1]
        if diff_i <= diff:
            diff = diff_i
            header_idx = i-1
    return header_idx

def extract_img_rows(img_raw, cutoff):

    """
        Take in a single image as a numpy array and extract the rows
        corresponding to property records.
    """

    row_starts=get_row_starts(img_raw=img_raw, cutoff=cutoff)
    img_rows=crop_img_rows(img_raw=img_raw, row_starts=row_starts)
    
    return row_starts, img_rows

def get_col_starts(header_row_raw, cutoff, window_width=100, lpad=100,
                   use_row_midsection=False):
    """
        Do a moving n-pixel wide moving average of the header's
        intensity to determine when a new column of data has begun.
    """

    # Specifiy whether you should only consider the midsection of the 
    # array when determinint the means, thus avoiding potential text
    # overhanging from the line above that could make it seem like
    # there was text
    if use_row_midsection == False:
        header_row_mean = np.mean(np.array(header_row_raw), 0)
    else:
        vertical_crop=np.floor(header_row_raw.shape[0]*0.1).astype(int)
        header_row_mean = np.mean(np.array(header_row_raw)[vertical_crop:-vertical_crop, :], 0)
    
    col_start_l = []

    prev_window_mean = cutoff

    for i in range(lpad+1, len(header_row_mean)):
        window_start = i-window_width
        window_end = i
        window_mean = np.mean(header_row_mean[window_start:window_end])
        if window_mean < cutoff and prev_window_mean >= cutoff:
            col_start_l.append(window_end)
        prev_window_mean = window_mean
    
    return col_start_l

def tighten_img_crop(row_col_cell):

    """
        Take in a single cell and tighten the crop 
        by eliminating empty whitespace.
    """

    # Crop pixel columns
    right_crop_idx = row_col_cell.shape[1]
    col_mean=np.mean(row_col_cell,0)
    for i in range(len(col_mean)-1,0,-1):
        if col_mean[i] != 255.0:
            right_crop_idx = i + 2
            right_crop_idx = np.amin([right_crop_idx, row_col_cell.shape[1]])
            break

    left_crop_idx = 0
    col_mean=np.mean(row_col_cell,0)
    for i in range(len(col_mean)):
        if col_mean[i] != 255.0:
            left_crop_idx = i - 2
            left_crop_idx = np.amax([left_crop_idx, 0])
            break   
    
    # Crop pixel rows
    bottom_crop_idx = row_col_cell.shape[0]
    row_mean=np.mean(row_col_cell,1)
    for i in range(len(row_mean)-1,0,-1):
        if row_mean[i] != 255.0:
            bottom_crop_idx = i + 2
            bottom_crop_idx = np.amin([bottom_crop_idx, row_col_cell.shape[0]])
            break

    top_crop_idx = 0
    row_mean=np.mean(row_col_cell,1)
    for i in range(len(row_mean)):
        if row_mean[i] != 255.0:
            top_crop_idx = i - 2
            top_crop_idx = np.amax([top_crop_idx, 0])
            break

    # Ensure that all new crop indices are within the bounds of
    # the original image
    assert left_crop_idx >= 0
    assert top_crop_idx >= 0
    assert right_crop_idx <= row_col_cell.shape[1]
    assert bottom_crop_idx <= row_col_cell.shape[0]
    
    cropped_cell = row_col_cell[top_crop_idx:bottom_crop_idx, 
                                left_crop_idx:right_crop_idx]

    return cropped_cell

def get_line_starts(img_raw, cutoff):
    """
        Identify the darker pixel rows, which indicate a line
        Return the start of the rows
    """

    light_rows = np.where(np.mean(img_raw, 1) >= cutoff)[0]
    light_row_starts = np.ediff1d(light_rows, to_begin=light_rows[0]-1) - 1
    light_row_starts = light_row_starts>0
    return light_rows[light_row_starts]

def split_cell_into_lines(row_col_cell, cutoff=250):

    # Determine mean row pixel value to figure out where the line starts
    line_starts = get_line_starts(img_raw=row_col_cell, cutoff=cutoff)
    print(type(line_starts[0]))
    
    # Append the first and last rows of the row/col cell and sort
    line_starts=np.append(line_starts, [0, row_col_cell.shape[0]])
    line_starts.sort()
    print(line_starts)
    
    # Determine the distance between each of the line starts to determine
    # which are really valid
    line_starts_diffs = np.ediff1d(line_starts, to_begin=line_starts[0])
    print(line_starts_diffs)

    # Filter the diffs and the starts by those diffs that are too small
    line_starts = line_starts[line_starts_diffs>10]
    line_starts_diffs = line_starts_diffs[line_starts_diffs>10]

    # Calculate the median diff, and only keep those line starts
    # that are close to the median diff
    median_line_start_diff = np.median(line_starts_diffs)
    print(line_starts_diffs)
    print(line_starts)
    bool_mask=np.abs(line_starts_diffs-median_line_start_diff)<=10

    #TODO this logic needs to be reworked.
    for i in range(len(bool_mask)-1):
        if bool_mask[i] == False and bool_mask[i+1] == True:
            bool_mask[i] = True
    
    line_starts = line_starts[bool_mask]
    line_starts = np.append(line_starts, [0])
    line_starts.sort()

    # Split up row/cell by line starts

    line_dict = {}
    print('len(line_starts)-1 :',len(line_starts)-1)
    for i in range(len(line_starts)-1):
        start = line_starts[i]
        end = line_starts[i+1]
        end +=1
        key = "line"+str(i)
        print(key)
        line = row_col_cell[start:end, :]
        line_dict.update({key : line})

    return line_dict

def crop_img_cols(img_rows, col_starts, left_buffer):
    """
        Take in a numpy array and crop the array based on the 
        supplied array of indicies that determine the start
        of a row's new column.
    """

    row_l = []

    for r in range(len(img_rows)):          
        start_idx = 0
        stop_idx = 1

        row_r = img_rows[r]
        row_r_dict = {}

        for i in range(len(col_starts)):
            crop_start = col_starts[start_idx]-left_buffer
            
            if stop_idx > len(col_starts)-1:
                crop_end = row_r.shape[1]
            else:
                crop_end = col_starts[stop_idx]
            
            row_col_i = row_r[:,crop_start-1:crop_end-1]

            # Crop the row/col cell
            row_col_i = tighten_img_crop(row_col_cell=row_col_i)
            
            col_i_name = 'col'+str(i)

            row_r_dict.update({col_i_name : row_col_i})

            start_idx += 1
            stop_idx += 1
        
        row_l.append(row_r_dict)

    return row_l

def extract_row_cols(img_rows, row_starts):
    
    """
        Take in a list of rows, identify the column starts,
        crop the images based on those indices, and
        return a list of dictionaries
    """

    # Determine the header row
    header_idx=identify_header_row(row_starts=row_starts)

    # Determine the column starts based on the pixels of the header row
    # Add a buffer to cut out the horizontal line

    col_starts=get_col_starts(header_row_raw=img_rows[header_idx][40:-20,:], cutoff=255)

    # Return a list of dictionaries, representing a row and its columns
    img_cols = crop_img_cols(img_rows=img_rows, col_starts=col_starts, left_buffer=15)

    return img_cols

def remove_footer(img_raw):
    """
    Remove the footer from the page image. The footer is a standard size
    for each page
    """
    crop_height = np.floor(img_raw.shape[0]*0.955).astype(int)
    img_mod = img_raw[0:crop_height, :]
    return(img_mod)

def extract_img_rows_and_cols(pdf_sheet_dir, cutoff):

    """
        Take in a list of images and return:
          - a list of images composed of
            - a list of rows composed of 
              - a dictionary of columns
    """
    img_raw_l = []
    pdf_sheets = os.listdir(pdf_sheet_dir)
    pdf_sheets.sort()

    for i in range(len(pdf_sheets)):
        progress_msg = '\rReading in and converting JPEGs ('+str(i+1)+'/'+str(len(pdf_sheets))+')'
        sys.stdout.write(progress_msg)
        sys.stdout.flush()
        sheet_fn = os.path.join(pdf_sheet_dir, pdf_sheets[i])
        sheet = Image.open(sheet_fn).convert('L')
        sheet = np.array(sheet)
        img_raw_l.append(sheet)
    
    print('\n')
    
    image_l = []

    for i in range(len(img_raw_l)):
        progress_msg = '\rExtracting image rows and columns ('+str(i+1)+'/'+str(len(img_raw_l))+')'
        sys.stdout.write(progress_msg)
        sys.stdout.flush()
        img_raw_i = img_raw_l[i]
        img_raw_i = remove_footer(img_raw_i)
        row_starts, img_rows = extract_img_rows(img_raw_i, cutoff)

        img_rows = extract_row_cols(img_rows=img_rows, row_starts=row_starts)
        image_l.append(img_rows)

    return image_l

def get_char_contours(img):
    
    """
    Return the coutours of the images
    source: https://stackoverflow.com/questions/50777688/finding-contours-with-lines-of-text-in-opencv
    """

    ret,thresh = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_TC89_KCOS)

    return contours[0]

def get_bb_coord(contour, img, IMG_HEIGHT):

    """
    Take in a single countour array generated by cv2.findContours
    and return the four corners of a bounding box that covers
    all of the coordinates.
    """

    # Get the height of the array and reshape the array into a hx2 array
    h = contour.shape[0]
    c = np.reshape(contour, (h, 2))
    min = np.argmin(c, 0)
    max = np.argmax(c, 0)
    left = np.max([c[min[0], 0]-1, 0])
    top = np.max([c[min[1], 1]-1, 0])
    right = np.min([c[max[0], 0]+1, img.shape[1]])
    bottom = np.min([c[max[1], 1]+1, img.shape[0]])
    coords = (left, top, right, bottom)
    
    # Check for overlapping characters between lines
    if bottom-top > 1.2*IMG_HEIGHT:
        mid_point = np.floor((bottom-top)/2).astype(int)+top
        coord_top = (left, top, right, mid_point)
        coord_bottom = (left, mid_point, right, bottom)
        coords = [coord_top, coord_bottom]
    assert left < right
    assert top < bottom
    return coords

def make_bb_coord_l(contour_l, img, IMG_HEIGHT):
    
    """
    Take in a list of contour arrays and return a list of four coordinates
    of a bounding box for each contour array.
    """
    
    assert isinstance(contour_l, list)

    coord_l = []

    for i in range(len(contour_l)):
        c = contour_l[i]
        bb = get_bb_coord(contour=c, img=img, IMG_HEIGHT=IMG_HEIGHT)
        
        # extend if bb is a list (i.e. a split bounding box)
        if isinstance(bb, list):
            coord_l.extend(bb)
        else:
            coord_l.append(bb)
    
    return coord_l

def inspect_bb_coords(bb_coords):
    None

def char_max_height(bb_coords, IMG_HEIGHT):
    """
    Take a list of bounding box coordinates and determine the greatest height
    among all of the boxes
    """
    l = []

    for i in range(len(bb_coords)):
        c = bb_coords[i]
        c_top, c_bottom = c[1], c[3]
        c_height = c_bottom - c_top
        l.append(c_height)
    
    l.sort()
    
    max_height = np.amin([l[-1], IMG_HEIGHT])

    return max_height

def char_max_width(bb_coords, IMG_WIDTH):
    """
    Take a list of bounding box coordinates and determine the greatest height
    among all of the boxes
    """
    l = []

    for i in range(len(bb_coords)):
        c = bb_coords[i]
        c_left, c_right = c[0], c[2]
        c_width = c_right - c_left
        l.append(c_width)
    
    l.sort()
    
    max_width = np.amin([l[-1], IMG_WIDTH])

    return max_width

def resize_img_letter(letter_img, img_height, img_width):

    """
    Get the image to as close to the specified dimensions as possible
    by adding/subtracting whitespace along the borders.
    """

    letter_img_h, letter_img_w = letter_img.shape[0], letter_img.shape[1]

    letter_img_mod = letter_img

    if letter_img_h > img_height:
        nrows_to_remove = letter_img_h - img_height
        nrows_to_remove_top = int(np.floor(nrows_to_remove/2))
        nrows_to_remove_bottom = int(np.ceil(nrows_to_remove/2))
        letter_img_mod = letter_img_mod[nrows_to_remove_top:(-1*nrows_to_remove_bottom), :]
    elif letter_img_h < img_height:
        nrows_to_add = img_height - letter_img_h
        nrows_to_add_top = int(np.floor(nrows_to_add/2))
        nrows_to_add_bottom = int(np.ceil(nrows_to_add/2))
        letter_img_mod = np.concatenate((np.full(shape=(nrows_to_add_top, letter_img_w), fill_value=255.0),
                                   letter_img_mod,
                                   np.full(shape=(nrows_to_add_bottom, letter_img_w), fill_value=255.0)),
                                  axis = 0)

    if letter_img_w > img_width:
        ncols_to_remove = letter_img_w - img_width
        ncols_to_remove_left = int(np.floor(ncols_to_remove/2))
        ncols_to_remove_right = int(np.ceil(ncols_to_remove/2))
        letter_img_mod = letter_img_mod[:, ncols_to_remove_left:(-1*ncols_to_remove_right)]
    elif letter_img_w < img_width:
        ncols_to_add = img_width - letter_img_w
        ncols_to_add_left = int(np.floor(ncols_to_add/2))
        ncols_to_add_right = int(np.ceil(ncols_to_add/2))
        letter_img_mod = np.concatenate((np.full(shape=(img_height, ncols_to_add_left), fill_value=255.0),
                                   letter_img_mod,
                                   np.full(shape=(img_height, ncols_to_add_right), fill_value=255.0)),
                                  axis = 1)

    return letter_img_mod

def sort_bb_coords(bb_coords, img, max_height, max_width, 
                   IMG_HEIGHT, dist_coords=(0,0)):
    """
    Assign all of the bb_coords to a list, unassigned_bb.

    Identify the bounding box that is closest to the upper lefthand corner
    of the frame. This is the first letter.

    Separate out the bounding boxes where the top coordinate is above the bottom
    coordinate of the first letter into a list line_0. Also remove these 
    bounding boxes from unassigned_bb. This should identify all of the letters 
    that are in the same line as the first letter.

    Sort each bounding box in line_0 by the lefthand coordinate to order the 
    letters correctly.

    Repeat this process until all unassigned_bb have been assigned to a 
    line list.
    """

    # Calculate distance from each bounding box's top-left corner to the
    # specified coordinates. This parameter is modified for cells
    # that with text that isn't left aligned. Otherwise, some rows of text
    # might be missed
    distance_l = []
    for i in range(len(bb_coords)):
        d = np.sqrt(np.power(np.abs(bb_coords[i][0]-dist_coords[0]),2) + 
                    np.power(np.abs(bb_coords[i][1]-dist_coords[1]),2))
        distance_l.append(d)

    # Create a list of coordinate/distance dictionaries
    unassigned_bb = []
    for i in range(len(bb_coords)):
        coord_dict = {"coord":bb_coords[i], 'dist':distance_l[i]}
        unassigned_bb.append(coord_dict)
    # Initialize the dictionary to store all of the lines
    line_dict = {}
    key_idx=0
    counter = 0

    while len(unassigned_bb) > 0:
        counter += 1
        if counter >= 100:
            break
        # Identify which of the remaining items in unassigned_bb has the 
        # shortest distance to (0,0)
        shortest_dist = np.Inf
        shortest_dist_idx = None

        for i in range(len(unassigned_bb)):
            dist = unassigned_bb[i].get('dist')
            if dist < shortest_dist:
                shortest_dist = dist
                shortest_dist_idx = i

        first_letter = unassigned_bb[shortest_dist_idx].get('coord')
        
        line_i = []
        remove_idxs = []

        # Determine which bounding boxes are in the same line as the first 
        # letter
        # Get boxes in line 1
        for i in range(len(unassigned_bb)):
            coord = unassigned_bb[i].get("coord")
            # If the top coordinate (1) of the unassigned bb is less than the 
            # bottom coordinate (3) of the first bb (i.e. 'above' in the image)
            # then append it to the list

            if coord[1] <= first_letter[3]:
                line_i.append(coord)
                remove_idxs.append(i)

        # Sort and reverse the list to remove items going from the back of 
        # the list to the front.
        remove_idxs.sort()
        remove_idxs.reverse()

        # Remove boxes in line 1 from unassigned_bb
        for i in remove_idxs:
            #print(i)
            unassigned_bb.pop(i)
        
        # Sort the bounding boxes by the lefthand coordinate (0) in 
        # a data frame.
        line_df = pd.DataFrame(np.array(line_i))
        line_df = line_df.sort_values(by=0)

        if line_df.shape[0] < 1:
            continue
        
        # Get the coordinates that cover the whole line
        # - The smallest lefthand and top coordinates plus 
        #   the largest righthand and bottom coordinates
        whole_line_bb = (line_df.iloc[:,0].min(), line_df.iloc[:,1].min(),
                         line_df.iloc[:,2].max(), line_df.iloc[:,3].max())
        whole_line_array = img[whole_line_bb[1]:whole_line_bb[3],
                               whole_line_bb[0]:whole_line_bb[2]]
        line_word_starts = get_col_starts(whole_line_array, cutoff=255, 
                                          window_width=40,lpad=0, 
                                          use_row_midsection=True)
        line_word_starts.append(0)

        # Append the largest lower boundary coordinate (i.e. the lowest on the page)
        # so that the final stop_idx has a value
        line_word_starts.append(line_df.iloc[:,2].max())
        line_word_starts.sort()

        # Crop each word in each line and append to a list
        line_words = []
        for i in range(len(line_word_starts)-1):
            start_idx = line_word_starts[i]
            stop_idx = line_word_starts[i+1]
            line_word = tighten_img_crop(whole_line_array[:, start_idx:stop_idx-1])
            line_words.append(line_word)

        # Extract each letter from each word in each line and maintain sort
        # Data structure: [[A,l,a,n], [B,r,o,d,y]]
        
        line_words_letters = []
        for i in range(len(line_words)):
            word_i = line_words[i]
            contour_l = get_char_contours(img=word_i)
            bb_coord_l = make_bb_coord_l(contour_l=contour_l, img=word_i,
                                         IMG_HEIGHT=IMG_HEIGHT)
            bb_array = np.array(bb_coord_l)
            if bb_array.shape[0] < 1:
                continue
            bb_array=bb_array[np.where((abs(bb_array[:,0]-bb_array[:,2])*abs(bb_array[:,1]-bb_array[:,3])>= 900))]
            bb_array=pd.DataFrame(bb_array)
            bb_array=bb_array.sort_values(by=0)
            line_word_i_letters = []

            for i in range(bb_array.shape[0]):
                line_word_i_letters_i = resize_img_letter(
                    letter_img=word_i[bb_array.iloc[i,1]:bb_array.iloc[i,3],
                           bb_array.iloc[i,0]:bb_array.iloc[i,2]], 
                    img_height=IMG_HEIGHT, img_width=50) #fixed
                line_word_i_letters.append(line_word_i_letters_i)
            line_words_letters.append(line_word_i_letters)

        cropped_list = []

        # Crop each bounding box and append to list in the same order
        for i in range(line_df.shape[0]):
            bb = tuple(line_df.iloc[i,:])
            # Append a cropped image array to the list
            bb = crop_text(img=img, max_height=max_height, 
                           max_width=max_width, bb=bb)
            cropped_list.append(bb)

        key = 'line'+str(key_idx)
        line_dict.update({key:cropped_list})
        key2 = 'whole_line'+str(key_idx)
        line_dict.update({key2:whole_line_array})
        key3 = 'line_words'+str(key_idx)
        line_dict.update({key3:line_words})
        key4 = 'line_words_letters'+str(key_idx)
        line_dict.update({key4:line_words_letters})
        key_idx += 1

    return line_dict

def crop_text(img, max_height, max_width, bb):
    """
    Take in one bounding box and a numpy array representing an images and 
    crop the text within the bounding box to a maximum height 
    """
    bb_top, bb_bottom, bb_left, bb_right = bb[1], bb[3], bb[0], bb[2]
    bb_top = bb_bottom - max_height
    bb_right = bb_left + max_width
    if bb_top <= 0:
        bb_top = 0
    if bb_right >= img.shape[1]:
        bb_right = img.shape[1]
    
    crop_section = img[bb_top:bb_bottom, bb_left:bb_right]

    return crop_section

def extract_and_sort_chars(img, IMG_HEIGHT, IMG_WIDTH, dist_coords):
    
    """
    Take in a row-col cell and return a dictionary where each key is a
    list of images, corresponding to the lines of text in the image.
    """

    contour_l = get_char_contours(img=img)
    bb_coord_l = make_bb_coord_l(contour_l=contour_l, img=img, IMG_HEIGHT=IMG_HEIGHT)
    max_height = np.amin([char_max_height(bb_coords=bb_coord_l, IMG_HEIGHT=IMG_HEIGHT),
                          IMG_HEIGHT])
    max_width = np.amin([char_max_width(bb_coords=bb_coord_l, IMG_WIDTH=IMG_WIDTH),
                         IMG_WIDTH])
    sorted_bb_coords = sort_bb_coords(bb_coords=bb_coord_l, img=img, 
                            max_height=max_height, max_width=max_width, 
                            IMG_HEIGHT=IMG_HEIGHT,
                            dist_coords=dist_coords)
    
    return sorted_bb_coords

def prep_extracted_img_for_pred(extracted_img, IMG_HEIGHT, IMG_WIDTH):
    array_from_img = tf.keras.preprocessing.image.img_to_array(extracted_img)
    array_from_img = np.array(Image.fromarray(extracted_img).convert('RGB'))
    array_from_img = tf.expand_dims(array_from_img, 0)

    array_from_img = tf.image.resize(array_from_img, size = [IMG_HEIGHT, IMG_WIDTH])
    
    return array_from_img

def assemble_pred_batch(img_list, IMG_HEIGHT, IMG_WIDTH):

    """
    Take in a list of image arrays, convert them to the formate
    expected by the model, and makes a dataset from the list.
    """

    batch_l = []

    for img_array in img_list:
        batch_l.append(prep_extracted_img_for_pred(
            extracted_img=img_array, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH
            ))
    
    batch_dataset=tf.data.Dataset.from_tensor_slices(batch_l)

    return batch_dataset

def get_image_text(processed_imgs, IMG_HEIGHT, IMG_WIDTH, model, labels):

    """
    Take in a list of processed images generated by extract_img_rows_and_cols.
    Return
    """
    # Ensure that the model and the labels have the same number of outputs
    # Otherwise, they might be from separate experiments
    assert model.outputs[0].shape[1]==labels.shape[0]

    doc_text = []
    for sheet_idx, sheet in enumerate(processed_imgs):
        progress_msg = '\rExtracting image text ('+str(sheet_idx+1)+'/'+str(len(processed_imgs))+')'
        sys.stdout.write(progress_msg)
        sys.stdout.flush()
        
        sheet_text = []
        for idx, row in enumerate(sheet):
            #print('len(sheet):')
            if idx==0:
                continue
            row_text = []
            for key, cell in row.items():
                #display(Image.fromarray(cell))
                if np.mean(cell) >= 254.0:
                    continue
                key_text_l=[]
                cell_img = Image.fromarray(cell)
                #display(cell_img)
                if key == 'col2':
                    dist_coords=(300,0)
                else:
                    dist_coords=(0,0)

                sorted_text_img = extract_and_sort_chars(
                    img=cell, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, 
                    dist_coords=dist_coords
                )
                keys_df=pd.DataFrame({'key':list(sorted_text_img.keys())})
                keys_df=keys_df.loc[keys_df.key.str.match(pat='line_words_letters\d')]
                #print(keys_df)
                
                for k in keys_df.key:
                    #print(k)
                    line_words_img_l = sorted_text_img.get(k)
                    #print(len(line_words_img_l))
                    line_words_txt_l = []
                    for w in range(len(line_words_img_l)):
                        word_img_l = line_words_img_l[w]
                        letters_txt_l = []
                        word_img_batch = assemble_pred_batch(word_img_l, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH)
                        if len(word_img_batch) == 0:
                            continue
                        preds = model.predict(word_img_batch)
                        scores = tf.nn.softmax(preds)
                        pred_labels=labels.class_names.str.extract('(\w$)').loc[np.argmax(scores, axis=1)]
                        pred_concat=''.join(list((pred_labels.iloc[:,0])))
                        line_words_txt_l.append(pred_concat)
                    key_text_l.append(line_words_txt_l)
                row_text.append(key_text_l)
            sheet_text.append(row_text)
        doc_text.append(sheet_text)
    return doc_text

def reorganize_doc_text(doc_text):
    col_list = []
    for sheet in doc_text:
        for row in sheet:
            row_dict = {}
            for c in range(len(row)):
                col = row[c]
                # col0: Name and Address of Last Reputed Owner
                n_lines = len(col)
                if c == 0 and n_lines > 1:
                    col0={}
                    # Determine if there is a address in this column or if there
                    # are only owner names
                    city_state_zip_idx = None
                    
                    for i, line in enumerate(col):
                        concat_line = ' '.join(line)
                        if re.search('\w{2}\s+\w{5}$', concat_line)!=None:
                            city_state_zip_idx = i
                    
                    # If there is an address line, extract the components
                    if city_state_zip_idx != None:
                        city_state_zip = col[city_state_zip_idx].copy()
                        city_state_zip_full = ' '.join(col[city_state_zip_idx].copy())
                        state_zip_set = set()
                        for word_i in city_state_zip:
                            if re.search(pattern='(\w{5}$)', string=word_i)!=None:
                                col0.update({'owner_zip':word_i})
                                state_zip_set.add(word_i)
                            elif re.search(pattern='^\w{2}$', string=word_i)!=None:
                                col0.update({'owner_state':word_i})
                                state_zip_set.add(word_i)

                        # Assume anything else in this line is the city
                        city = []
                        for i in city_state_zip:
                            if 'owner_state' and 'owner_zip' in col0.keys() and i not in [col0.get('owner_state'),col0.get('owner_zip')]:
                                city.append(i)                  
                        col0.update({'owner_city':' '.join(city)})
                        # Assume the line immediately above is the street address
                        col0.update({'owner_street_addr':' '.join(col[city_state_zip_idx-1])})
                        owner_line_end = city_state_zip_idx-1
                    else: 
                        owner_line_end=len(col)
    
                    for i, owner in enumerate(col[0:owner_line_end]):
                        key = 'owner_name'+str(i+1)   
                        val = ' '.join(owner)
                        col0.update({key:val})   
                    row_dict.update({'col0':col0})
                # col1: County tax map, address, and account identifier
                if c == 1 and n_lines > 1:
                    col1={}
                    
                    for i, line in enumerate(col):
                        concat_line = ' '.join(line)
                        if re.search(pattern='ITEM\s+NO', string=concat_line, flags=re.IGNORECASE)!=None:
                            item_no_idx = i
                            col1.update({'item_no': re.findall(pattern = '\w+$', string = concat_line)[0]})
                        if re.search(pattern='\w{6}\s+\w{16}', string=concat_line)!=None:
                            account_idx = i
                            col1.update({'account': concat_line})
                        if re.search(pattern='TAX\s+CODE', string=concat_line, flags=re.IGNORECASE)!=None:
                            tax_code_idx = i
                    addr_l = []
                    for i, line in enumerate(col):
                        concat_line = ' '.join(line)
                        if i not in [item_no_idx, account_idx, tax_code_idx]:
                            addr_l.append(concat_line)
                    addr = ' '.join(addr_l)
                    col1.update({'property_addr':addr})
                    row_dict.update({'col1':col1})
                
                # col2: Assessed land/Assessed total/market value
                if c == 2 and n_lines > 1:
                    col2={}
                    for i, line in enumerate(col):
                        concat_line = ' '.join(line)
                        if i == 0:
                            col2.update({'assessed_land':concat_line})
                        if i == 1:
                            col2.update({'assessed_total':concat_line})
                        if i == 2:
                            col2.update({'total_market_value':concat_line})
                        if re.search(pattern='^SCH', string=concat_line)!=None:
                            col2.update({'school_no': re.findall(pattern = '\w+$', string = concat_line)[0]})
                        if re.search(pattern='^CLASS', string=concat_line)!=None:
                            col2.update({'class_no': re.findall(pattern = '\w+$', string = concat_line)[0]})
                        if re.search(pattern='^ACRES', string=concat_line)!=None:
                            col2.update({'acres': re.findall(pattern = '\w+$', string = concat_line)[0]})
                    row_dict.update({'col2':col2})
                
                # col5: Net taxable
                if c == 5 and n_lines > 1:
                    col5={}
                    for i, line in enumerate(col):
                        concat_line = ' '.join(line)
                        if re.search(pattern='^County', string=concat_line, flags=re.IGNORECASE)!=None:
                            col5.update({'net_taxable_county':re.findall(pattern = '\w+$', string = concat_line)[0]})
                        elif re.search(pattern='^Town', string=concat_line, flags=re.IGNORECASE)!=None:
                            col5.update({'net_taxable_town':re.findall(pattern = '\w+$', string = concat_line)[0]})                    
                        else:
                            col5.update({'net_taxable_school':re.findall(pattern = '\w+$', string = concat_line)[0]})
                    row_dict.update({'col5':col5})

            col_list.append(row_dict)
    return col_list

def doc_text_to_dataframe(reorganized_doc_text):
    flat_df_l = []
    for i, row in enumerate(reorganized_doc_text):
        flat_col_l = []
        for key, col in row.items():
            df = pd.DataFrame(col, index=[0])
            flat_col_l.append(df)
        if len(flat_col_l)>=1:
            flat_df = pd.concat(flat_col_l, axis=1)
            flat_df_l.append(flat_df)
    combined_df = pd.concat(flat_df_l, axis=0, ignore_index=True)
    combined_df=combined_df[['owner_name1','owner_name2',
                            'owner_name3','owner_name4',
                            'owner_street_addr','owner_city',
                            'owner_state','owner_zip',
                            'account','item_no','property_addr',
                            'assessed_land','assessed_total','total_market_value',
                            'school_no','class_no','acres',
                            'net_taxable_county','net_taxable_town',
                            'net_taxable_school']]
    return combined_df
