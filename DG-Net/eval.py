import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--UIE', action='store_true', help='output UIE dataset')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')

eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')

ep = eval_parser.parse_args()


def eval(model, testing_data_loader, model_path, output_folder, LOL=False):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded: ', model_path)
    model.eval()
    print('Evaluation:')
    # if LOL:
    #     model.trans.gated = True
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            # 开始推理
            input, input_wb, input_gc, input_his, file, file_wb, file_gc, file_his = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
            input = input.cuda()
            input_wb = input_wb.cuda()
            input_gc = input_gc.cuda()
            input_his = input_his.cuda()
            # print(input.shape)
            # print(input_wb.shape)
            # print(input_gc.shape)
            # print(input_his.shape)
            output, _, output_raw= model(input, input_wb, input_gc, input_his)
            # print(output.shape)
            
        if not os.path.exists(output_folder):          
            os.makedirs(output_folder)  

        output = torch.clamp(output.cuda(),0,1).cuda()
        # 保存图像
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + file[0])
        
        # 保存 output_raw
        raw_folder = os.path.join(output_folder, "raw")
        os.makedirs(raw_folder, exist_ok=True)  # 如果不存在就创建
        output_raw = torch.clamp(output_raw.cuda(),0,1).cuda()
        raw_img = transforms.ToPILImage()(output_raw.squeeze(0))
        raw_img.save(os.path.join(raw_folder, file[0]))

        torch.cuda.empty_cache()
    print('===> End evaluation')
    # if LOL:
    #     model.trans.gated = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.UIE:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/780/eval/raw-780"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/raw-780/'
        if ep.perc:
            weight_path = './weights/UIE/w_perc.pth' # 自己模型训练得到的权重
        else:
            weight_path = './weights/UIE/wo_perc.pth'

    eval_net = CIDNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol)

