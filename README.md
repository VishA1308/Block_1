# Block_1
## Основные функции

#Разбиение изображения на блоки:

blocks, width, height = split_image_into_blocks(image_path, block_size)


#Обработка блоков (усреднение цветов по каналам и создание блока):

colored_blocks = split_img_into_canals(blocks)


#Восстановление изображения из блоков:

reconstructed_img = reconstruct_image(colored_blocks, width, height, block_size)
