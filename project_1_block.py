import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import keyboard

def split_image_into_blocks(image_path, block_size):
    """
    Function, which drop image for blocks 16x16
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось открыть изображение: {image_path}")
        return None, None, None

    if block_size <= 0:
        print("Ошибка: Размер блоков должен быть положительным числом")
        return None, None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    blocks = []
    
    #print(f"Размер изображения: {width}x{height}")
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            
            block = img[y:y_end, x:x_end]
            
            blocks.append(block)
    
    return blocks, width, height

def split_img_into_canals(blocks):
    """
    In every block find median for channels
    """
    colored_blocks = []
    
    for i, block in enumerate(blocks):
        r, g, b = cv2.split(block)  
        
        r_mean = round(np.mean(r))
        g_mean = round(np.mean(g))
        b_mean = round(np.mean(b))
        
        colored_block = np.full_like(block, [r_mean, g_mean, b_mean])
        colored_blocks.append(colored_block)
    
    return colored_blocks

def reconstruct_image(blocks, original_width, original_height, block_size):

    reconstructed = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    
    block_index = 0
    for y in range(0, original_height, block_size):
        for x in range(0, original_width, block_size):
            y_end = min(y + block_size, original_height)
            x_end = min(x + block_size, original_width)
            
            
            block = blocks[block_index]
            reconstructed[y:y_end, x:x_end] = block[:y_end-y, :x_end-x]
            
            block_index += 1
    
    return reconstructed

def show_blocks_cv2(blocks):
    j = 0
    total_blocks = len(blocks)
    
    while j < total_blocks:
        if keyboard.is_pressed('a'):

            
            block_bgr = cv2.cvtColor(blocks[j], cv2.COLOR_RGB2BGR)
            cv2.imshow(f'Block {j+1}/{total_blocks}', block_bgr)
            
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            j += 1
        
        if keyboard.is_pressed('q'):
            break
        
        time.sleep(0.01)


image_path = 'cat1.jpg'
block_size = 16


blocks, original_width, original_height = split_image_into_blocks(image_path, block_size)
if (blocks == None and original_width == None and original_height == None ):
    exit()
print(f"Получено блоков: {len(blocks)}")
print(f"Оригинальный размер: {original_width}x{original_height}")

if blocks:

    colored_blocks = split_img_into_canals(blocks)
    print(f"Обработано блоков: {len(colored_blocks)}")

   
    reconstructed_img = reconstruct_image(colored_blocks, original_width, original_height, block_size)
    

    original_img = cv2.imread(image_path)
    
   
    reconstructed_bgr = cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR)

    cv2.imshow('Original Image', original_img)
    cv2.imshow('Reconstructed Image ', reconstructed_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    cv2.imwrite('reconstructed_image.jpg', reconstructed_bgr)

    

