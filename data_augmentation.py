# Data Augmentation
import cv2
from PIL import Image, ImageEnhance
import time
import numpy as np

class Image_enhance():
    def __init__(self,rootPath):
    	
        #:param rootPath: input path
        self.rootPath = rootPath
        self.export_path_base = rootPath[:-4] 
        self.image = cv2.imread(rootPath)
        self.class_name = rootPath.split("\\")[-2]

    def get_savename(self,operate_name):

        try:

            now = time.time()
            tail_time = str(round(now * 1000000))[-4:]  
            head_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
            label = str(head_time + tail_time) + '_' + str(operate_name)

            export_path_base = self.export_path_base
          
            out_path = export_path_base
            # if not os.path.exists(out_path):
            #     os.mkdir(out_path)


            savename = out_path + '_' + label + ".jpg"

            return savename

        except Exception as e:
            print(e)


    def SaltAndPepper(self, percetage=0.2):
        """SaltAndPepper Noise"""
        SP_NoiseImg = self.image.copy()
        SP_NoiseNum = int(percetage * self.image.shape[0] * self.image.shape[1])
        for i in range(SP_NoiseNum):
            randR = np.random.randint(0, self.image.shape[0] - 1)
            randG = np.random.randint(0, self.image.shape[1] - 1)
            randB = np.random.randint(0, 3)
            if np.random.randint(0, 1) == 0:
                SP_NoiseImg[randR, randG, randB] = 0
            else:
                SP_NoiseImg[randR, randG, randB] = 255
        percetage_name = str(percetage*100).replace('.','')
        operate_name = 'SaltAndPepper_' + percetage_name
        save_name = self.get_savename(operate_name)
        cv2.imwrite(save_name,SP_NoiseImg)



    def addGaussianNoise(self, percetage=0.2):
        """addGaussianNoise"""
        G_Noiseimg = self.image.copy()
        w = self.image.shape[1]
        h = self.image.shape[0]
        G_NoiseNum = int(percetage * self.image.shape[0] * self.image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, h)
            temp_y = np.random.randint(0, w)
            G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
        percetage_name = str(percetage*100).replace('.','')
        operate_name = 'addGaussianNoise_' + percetage_name
        save_name = self.get_savename(operate_name)
        cv2.imwrite(save_name,G_Noiseimg)

    def darker(self, percetage=0.87):
        """darken the pictures"""
        image_darker = self.image.copy()
        w = self.image.shape[1]
        h = self.image.shape[0]
        # get darker
        for xi in range(0, w):
            for xj in range(0, h):
                image_darker[xj, xi, 0] = int(self.image[xj, xi, 0] * percetage)
                image_darker[xj, xi, 1] = int(self.image[xj, xi, 1] * percetage)
                image_darker[xj, xi, 2] = int(self.image[xj, xi, 2] * percetage)
        percetage_name = str(percetage*100).replace('.','')
        operate_name = 'darker_' + percetage_name
        save_name = self.get_savename(operate_name)
        cv2.imwrite(save_name,image_darker)

    def brighter(self, percetage=1.07):
        """brighten the pictures"""
        image_brighter= self.image.copy()
        w = self.image.shape[1]
        h = self.image.shape[0]
        # get brighter
        for xi in range(0, w):
            for xj in range(0, h):
                image_brighter[xj, xi, 0] = np.clip(int(self.image[xj, xi, 0] * percetage), a_max=255, a_min=0)
                image_brighter[xj, xi, 1] = np.clip(int(self.image[xj, xi, 1] * percetage), a_max=255, a_min=0)
                image_brighter[xj, xi, 2] = np.clip(int(self.image[xj, xi, 2] * percetage), a_max=255, a_min=0)
        percetage_name = str(percetage*100).replace('.','')
        operate_name = 'brighter_' + percetage_name
        save_name = self.get_savename(operate_name)
        cv2.imwrite(save_name,image_brighter)

    def rotate(self, angle=15, center=None, scale=1.0):
        """rotate the pictures"""
        (h, w) = self.image.shape[:2]
        # If no rotation center is specified, the center of the image is set as the rotation center
        if center is None:
            center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, angle, scale)
        rotate_image = cv2.warpAffine(self.image.copy(), m, (w, h))
        angle_name = str(angle).replace('.','')
        operate_name = 'rotate_' + angle_name
        save_name = self.get_savename(operate_name)
        cv2.imwrite(save_name,rotate_image)
        return save_name


    def flip(self):
        """horiziontal flip"""
        flipped_image = np.fliplr(self.image.copy())
        operate_name = 'flip_'
        save_name = self.get_savename(operate_name)
        cv2.imwrite(save_name,flipped_image)

    def deform(self):
        """deform pictures"""
        try:
            operate = 'deform_'
            rootPath = self.rootPath

            with Image.open(rootPath) as image:
                w, h = image.size
                w = int(w)
                h = int(h)
                if not w == h:
                    out_ww = image.resize((int(w), int(w)))
                    operate_name_ww = operate + str(w)
                    savename_ww = self.get_savename(operate_name_ww)
                    out_ww.save(savename_ww, quality=100)
                    out_hh = image.resize((int(h), int(h)))
                    operate_name_hh = operate + str(h)
                    savename_hh = self.get_savename(operate_name_hh)
                    out_hh.save(savename_hh, quality=100)
                else:
                    pass
            # logger.info(operate)
        except Exception as e:
            # logger.error('ERROR %s', operate)
            # logger.error(e)
            print(e,"ERROR"+str(operate))

    def crop(self,choose):
        """cut the corners or center"""
        """:choose which operation"""
        try:
            operate = 'crop_'
            rootPath = self.rootPath

            with Image.open(rootPath) as image:
                w, h = image.size
                scale = 0.875
                ww = int(w * scale)
                hh = int(h * scale)
                x = y = 0

                if choose =='lu':
                    x_lu = x
                    y_lu = y
                    out_lu = image.crop((x_lu, y_lu, ww, hh))
                    operate_lu_name =operate + 'lu'
                    savename_lu = self.get_savename(operate_lu_name)
                    out_lu.save(savename_lu, quality=100)
                # logger.info(operate + '_lu')

                elif choose =='ld':
                    x_ld = int(x)
                    y_ld = int(y + (h - hh))
                    out_ld = image.crop((x_ld, y_ld, ww, hh))
                    operate_ld_name =operate + 'ld'
                    savename_ld = self.get_savename(operate_ld_name)
                    out_ld.save(savename_ld, quality=100)
                # logger.info(operate + '_ld')

                elif choose =='ru':
                    x_ru = int(x + (w - ww))
                    y_ru = int(y)
                    out_ru = image.crop((x_ru, y_ru, w, hh))
                    operate_ru_name =operate + 'ru'
                    savename_ru = self.get_savename(operate_ru_name)
                    out_ru.save(savename_ru, quality=100)
                # logger.info(operate + '_ru')

                elif choose == 'rd':
                    x_rd = int(x + (w - ww))
                    y_rd = int(y + (h - hh))
                    out_rd = image.crop((x_rd, y_rd, w, h))
                    operate_rd_name =operate + 'rd'
                    savename_rd = self.get_savename(operate_rd_name)
                    out_rd.save(savename_rd, quality=100)
                # logger.info(operate + '_rd')

                elif choose == 'ce':
                    x_ce = int(x + (w - ww) / 2)
                    y_ce = int(y + (h - hh) / 2)
                    out_ce = image.crop((x_ce, y_ce, ww, hh))
                    operate_ce_name =operate + 'center'
                    savename_ce = self.get_savename(operate_ce_name)
                    out_ce.save(savename_ce, quality=100)
                else:
                    xx = ['lu','ld','ru','rd','ce']
                    print('Error, please check if you choose valid operation'.format(xx))
        except Exception as e:
            # logger.error('ERROR %s', 1)
            # logger.error(e)
            print(e,"ERROR"+str(operate))

    def image_color(self):
        """
        change the color slightly
        """
        image = Image.open(self.rootPath)
        random_factor = np.random.randint(low=0, high=31) / 10.0  
        color_image = ImageEnhance.Color(image).enhance(random_factor)  
        random_factor = np.random.randint(low=10, high=21) / 10.0
        bright_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
        random_factor = np.random.randint(low=10, high=21) / 10.0
        contrast_image = ImageEnhance.Contrast(bright_image).enhance(random_factor)  
        random_factor = np.random.randint(low=0, high=31) / 10.0
        sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor) 
        operate_color_name = 'color_'
        savename_color = self.get_savename(operate_color_name)
        sharp_image.save(savename_color)
