# Data Augmentation
import cv2
from PIL import Image, ImageEnhance
import time
import numpy as np


class Image_enhance():
    def __init__(self,rootPath):
    	
        #:param rootPath: 图像输入路径
        self.rootPath = rootPath
        self.export_path_base = rootPath[:-4] #图像输出路径基
        self.image = cv2.imread(rootPath)
        self.class_name = rootPath.split("\\")[-2]

    def get_savename(self,operate_name):
        """
        :param operate_name: 图像使用的数据增强操作类名
        :return: 返回图像存储名
        """
        try:

            # 获取时间戳，用于区分图像
            now = time.time()
            tail_time = str(round(now * 1000000))[-4:]  # 时间戳尾数
            head_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
            # 时间标签
            label = str(head_time + tail_time) + '_' + str(operate_name)

            # 输出文件夹
            export_path_base = self.export_path_base
            # 子文件夹以“操作operate”命名
            out_path = export_path_base
            # 创建子文件夹
            # if not os.path.exists(out_path):
            #     os.mkdir(out_path)

            # 存储完整路径
            savename = out_path + '_' + label + ".jpg"

            return savename

        except Exception as e:
            print(e)


    def SaltAndPepper(self, percetage=0.2):
        """给图片增加椒盐噪声"""
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
        #print('{}做数据增强{} 完毕 '.format(self.class_name,operate_name))


    def addGaussianNoise(self, percetage=0.2):
        """给图片增加高斯噪声"""
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
        #print('{}做数据增强{} 完毕 '.format(self.class_name,operate_name))

    def darker(self, percetage=0.87):
        """减低图片像素，是图片变昏暗"""
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
        #print('{}做数据增强{} 完毕 '.format(self.class_name, operate_name))

    def brighter(self, percetage=1.07):
        """增强图片像素，使图片变亮"""
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
        #print('{}做数据增强{} 完毕 '.format(self.class_name, operate_name))

    def rotate(self, angle=15, center=None, scale=1.0):
        """按指定角度旋转"""
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
        """水平翻转."""
        flipped_image = np.fliplr(self.image.copy())
        operate_name = 'flip_'
        save_name = self.get_savename(operate_name)
        cv2.imwrite(save_name,flipped_image)

    def deform(self):
        """图像拉伸."""
        try:
            operate = 'deform_'
            # 图像完整路径
            rootPath = self.rootPath

            with Image.open(rootPath) as image:
                w, h = image.size
                w = int(w)
                h = int(h)
                if not w == h:
                    # 拉伸成宽为w的正方形
                    out_ww = image.resize((int(w), int(w)))
                    operate_name_ww = operate + str(w)
                    savename_ww = self.get_savename(operate_name_ww)
                    out_ww.save(savename_ww, quality=100)
                    # 拉伸成宽为h的正方形
                    out_hh = image.resize((int(h), int(h)))
                    operate_name_hh = operate + str(h)
                    savename_hh = self.get_savename(operate_name_hh)
                    out_hh.save(savename_hh, quality=100)
                else:
                    pass

            # 日志
            # logger.info(operate)
        except Exception as e:
            # logger.error('ERROR %s', operate)
            # logger.error(e)
            print(e,"ERROR"+str(operate))

    def crop(self,choose):
        """提取四个角落和中心区域."""
        """:choose 指选择哪种操作，共可以选择五种切割操作"""
        try:
            operate = 'crop_'
            # 图像完整路径
            rootPath = self.rootPath

            with Image.open(rootPath) as image:
                w, h = image.size
                # 切割后尺寸
                scale = 0.875
                # 切割后长宽
                ww = int(w * scale)
                hh = int(h * scale)
                # 图像起点，左上角坐标
                x = y = 0

                # 切割左上角
                if choose =='lu':
                    x_lu = x
                    y_lu = y
                    out_lu = image.crop((x_lu, y_lu, ww, hh))
                    operate_lu_name =operate + 'lu'
                    savename_lu = self.get_savename(operate_lu_name)
                    out_lu.save(savename_lu, quality=100)
                # logger.info(operate + '_lu')

                # 切割左下角
                elif choose =='ld':
                    x_ld = int(x)
                    y_ld = int(y + (h - hh))
                    out_ld = image.crop((x_ld, y_ld, ww, hh))
                    operate_ld_name =operate + 'ld'
                    savename_ld = self.get_savename(operate_ld_name)
                    out_ld.save(savename_ld, quality=100)
                # logger.info(operate + '_ld')

                # 切割右上角
                elif choose =='ru':
                    x_ru = int(x + (w - ww))
                    y_ru = int(y)
                    out_ru = image.crop((x_ru, y_ru, w, hh))
                    operate_ru_name =operate + 'ru'
                    savename_ru = self.get_savename(operate_ru_name)
                    out_ru.save(savename_ru, quality=100)
                # logger.info(operate + '_ru')

                # 切割右下角
                elif choose == 'rd':
                    x_rd = int(x + (w - ww))
                    y_rd = int(y + (h - hh))
                    out_rd = image.crop((x_rd, y_rd, w, h))
                    operate_rd_name =operate + 'rd'
                    savename_rd = self.get_savename(operate_rd_name)
                    out_rd.save(savename_rd, quality=100)
                # logger.info(operate + '_rd')

                # 切割中心
                elif choose == 'ce':
                    x_ce = int(x + (w - ww) / 2)
                    y_ce = int(y + (h - hh) / 2)
                    out_ce = image.crop((x_ce, y_ce, ww, hh))
                    operate_ce_name =operate + 'center'
                    savename_ce = self.get_savename(operate_ce_name)
                    out_ce.save(savename_ce, quality=100)
                else:
                    xx = ['lu','ld','ru','rd','ce']
                    print('未剪切成功，请检查choose选择剪切的参数是否为{}中的一个'.format(xx))
                # logger.info('提取中心')
        except Exception as e:
            # logger.error('ERROR %s', 1)
            # logger.error(e)
            print(e,"ERROR"+str(operate))

    def image_color(self):
        """
        对图像进行颜色抖动
        """
        image = Image.open(self.rootPath)
        random_factor = np.random.randint(low=0, high=31) / 10.0  # 随机的扰动因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(low=10, high=21) / 10.0
        bright_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(low=10, high=21) / 10.0
        contrast_image = ImageEnhance.Contrast(bright_image).enhance(random_factor)  # 调整图像的对比度
        random_factor = np.random.randint(low=0, high=31) / 10.0
        sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像的锐度
        operate_color_name = 'color_'
        savename_color = self.get_savename(operate_color_name)
        sharp_image.save(savename_color)
