from UIQM import(
  getUIQM,
  eme_tf
)
import sys
import os
import cv2


def main():
    # 需要计算指标的图片所在路径
    result_path = sys.argv[1]
    result_dirs = os.listdir(result_path)

    N=0
    imagenum=1 # 目前处理图片是第几张
    for imgdir in result_dirs:
        # if '.png' in imgdir or '.jpg' in imgdir : # 文件夹内所有图片
        # if imgdir.endswith('.png') and not imgdir.endswith('io.png'):  # 文件夹内固定命名格式的图片
        # if  imgdir.endswith('original.png'): # 文件后缀
        if  imgdir.endswith('_out.png'): # 文件后缀
            #corrected image 注意与保存时一致（io读则io保存RGB，cv2读则cv2保存BGR）
            corrected = cv2.imread(os.path.join(result_path,imgdir))
            #图片名称为imgdir，此处提取前缀
            imgname_cor = imgdir.split('.')[0]
            #第几张：图片名称
            print(imagenum,":",imgname_cor)
            #计算指标
            uiqm = getUIQM(corrected)
            print(uiqm)
    
    
if __name__ == '__main__':
    main()