import glob
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque
from loss.niqe_utils import *
import argparse

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
ep = eval_parser.parse_args()


def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0
        
    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1
        
        im1 = Image.open(item).convert('RGB')
        # score_brisque = brisque.score(im1) 
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
        print(item, ": NIQE is ", score_niqe)
        
        
        # avg_brisque += score_brisque
        avg_niqe += score_niqe

        torch.cuda.empty_cache()
    
    # avg_brisque = avg_brisque / n
    avg_brisque = 0
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque

if __name__ == '__main__':

    # im_dir = './results_Weight2_MSE+5e-6VGG+12e-4FFL/results_150/res-C60/*.png'
    # im_dir = './results_RCG/results_MSEVGG_WB_5MSE+100000Edge+10color+125light+1e-4Vgg/res-C60/*.png'
    im_dir = './Test/Test-C60/*.png'
    
    avg_niqe, avg_brisque = metrics(im_dir)
    print("avg_niqe: ", avg_niqe)
    print("avg_brisque: ", avg_brisque)
#   2 :   4.552693115057218  5.354837018300595
# MSEVGG: 4.557825016379721  5.863802694452791
#  raw :  4.694433420395897  7.027499878936617