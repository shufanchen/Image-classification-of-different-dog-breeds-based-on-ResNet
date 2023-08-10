import os
from data_augmentation import Image_enhance

def augment_images_in_folders(base_folder):
    for breed_folder in os.listdir(base_folder):
        breed_path = os.path.join(base_folder, breed_folder)
        
        for image_file in os.listdir(breed_path):
            image_path = os.path.join(breed_path, image_file)
            
            image_augmentor = Image_enhance(image_path)
            image_augmentor.SaltAndPepper(percetage=0.2)
            image_augmentor.addGaussianNoise(percetage=0.2)
            image_augmentor.darker(percetage=0.87)
            image_augmentor.brighter(percetage=1.07)
            image_augmentor.rotate(angle=15)
            image_augmentor.flip()
            image_augmentor.deform()
            image_augmentor.crop(choose='lu')
            image_augmentor.crop(choose='ld')
            image_augmentor.crop(choose='ru')
            image_augmentor.crop(choose='rd')
            image_augmentor.crop(choose='ce')
            image_augmentor.image_color()
            
        print(breed_folder + '\t' + 'mission completed')

if __name__ == "__main__":
    # set the path 
    base_folder = 'train_'  

    augment_images_in_folders(base_folder)
