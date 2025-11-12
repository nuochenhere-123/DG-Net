# 修改-d0为测试图像文件夹，-d1为参考图像文件夹，-0未LPIPS单张图像记录文件
# 修改文件读取限制line28，并确保ref读取正确line32后，直接run
import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./img/Test/WO-SCA-110')
# parser.add_argument('-d0','--dir0', type=str, default='./img/Test/UShape-110-resize')
# parser.add_argument('-d1','--dir1', type=str, default='./img/Ref/110_reference_256')
parser.add_argument('-d1','--dir1', type=str, default='./img/Ref/UIEB-reference-890')
parser.add_argument('-o','--out', type=str, default='./img/Test/RetinexBased-110/WO-SCA-LPIPS-110.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
sum = 0
num = 0

for file in files:
    if file.endswith('_out.png') or file.endswith('_out.jpg') or file.endswith('_out.jpeg'):
    # if '.png' in file or '.jpg' in file  or '.jpeg' in file :  
               
        imgname = file.split('_out.')[0] # 以"."分为前后两部分 随机应变改！！！
        refdir = imgname+'.'+file.split('_out.')[1] # 参考图像名称（含后缀）随机应变改！！！ 
        # imgname = file.split('.')[0] # 以"."分为前后两部分 随机应变改！！！
        # refdir = imgname+'.'+file.split('.')[1] # 参考图像名称（含后缀）随机应变改！！！
        # refdir = imgname+'.'+'png' # 参考图像名称（含后缀）随机应变改！！！
        
        if os.path.exists(os.path.join(opt.dir1,refdir)):
            num += 1
            print("处理第", num, "张图像：", file)
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,refdir)))
            if(opt.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()
            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            sum += dist01
            print('%s: %.6f'%(file,dist01))
            f.writelines('%s: %.6f\n'%(file,dist01))

avg = sum/num
f.writelines('\nAVG: %.6f\n'%(avg))
print('AVG: %.6f'%(avg))

f.close()
