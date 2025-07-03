import os
import time
import argparse
import logging
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils.vgg import Vgg19
from datasets.datasets_pairs import my_dataset_eval
import matplotlib.image as img
import pyiqa
import sys
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('pyiqa').setLevel(logging.WARNING)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANS_EVAL = transforms.Compose([transforms.ToTensor()])

def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluation script for reflection removal models.")
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment/model.')
    parser.add_argument('--strategy', type=str, help='Training Strategy for the model.')
    parser.add_argument('--save_folder', type=str, default='/nfs.auto/AI/VideoSummarization/kangning/icip2025/Results', help='Path to save results.')
    parser.add_argument('--SAVE_test_Results', action='store_true', help='Flag to save test results.')
    parser.add_argument('--pre_model', type=str, required=True, help="Path to the pretrained model checkpoint.")
    parser.add_argument('--pre_model_Det', type=str, help="Path to the pretrained Detection model checkpoint for RDNet-RRNet.")
    # Dataset paths
    parser.add_argument('--eval_paths', type=str, nargs='+', help="List of dataset paths in the format: name,in_path,gt_path.")
    return parser.parse_args()

def load_model(args):
    """Load the appropriate model based on the experiment name."""
    if args.experiment_name == 'DSRNet':
        from DSRNetModels import arch
        net = arch.__dict__['dsrnet_l_nature'](3, 3).to(DEVICE)
        state_dict = torch.load(args.pre_model)
        net.load_state_dict(state_dict.get('weight', state_dict), strict=True)
        vgg = Vgg19(requires_grad=False).to(DEVICE)
        return net, vgg
    elif args.experiment_name == 'ERRNet':
        from ERRNetModels import arch
        net = arch.__dict__['errnet'](3 + 1472, 3).to(DEVICE)
        state_dict = torch.load(args.pre_model)
        net.load_state_dict(state_dict['icnn'], strict=True)
        vgg = Vgg19(requires_grad=False).to(DEVICE)
        return net, vgg
    elif args.experiment_name == 'RAGNet':
        from RAGNetModels.GT import GT_Model
        from RAGNetModels.GR import GR_Model
        from RAGNetModels.encoder_build import encoder
        encoder_I, encoder_R = encoder().to(DEVICE), encoder().to(DEVICE)
        gt_model = GT_Model(encoder_I, encoder_R).to(DEVICE)
        gr_model = GR_Model(encoder().to(DEVICE)).to(DEVICE)
        state_dict = torch.load(args.pre_model)
        gt_model.load_state_dict(state_dict['GT_state'])
        gr_model.load_state_dict(state_dict['GR_state'])
        return gt_model, gr_model
    elif args.experiment_name == 'RDNetRRNet':
        from RDNetRRNetModels.NAFNet_arch import NAFNet_wDetHead
        from RDNetRRNetModels.network_RefDet import RefDet
        net = NAFNet_wDetHead(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], concat=True).to(DEVICE)
        net_Det = RefDet(backbone='efficientnet-b3', proj_planes=16, pred_planes=32).to(DEVICE)
        net.load_state_dict(torch.load(args.pre_model), strict=True)
        net_Det.load_state_dict(torch.load(args.pre_model_Det), strict=True)
        return net, net_Det
    elif args.experiment_name == 'BaselineModel':
        from BaselineModels.RR_Network import NAFNet
        net = NAFNet(img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).to(DEVICE)
        state_dict = torch.load(args.pre_model)
        net.load_state_dict(state_dict['icnn'], strict=True)
        return net, None
    else:
        raise ValueError(f"Unknown experiment name: {args.experiment_name}")

def get_eval_loader(in_path, gt_path, transform=TRANS_EVAL):
    """Create a DataLoader for evaluation."""
    dataset = my_dataset_eval(root_in=in_path, root_label=gt_path, transform=transform, fix_sample=500)
    return DataLoader(dataset=dataset, batch_size=1, num_workers=4)

def test_model(net, net_Sup, eval_loader, dataset_name, save_results=False):
    """Evaluate the model on a given dataset."""
    with suppress_stdout():
        iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=False)
        iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=False)
        iqa_lpips = pyiqa.create_metric('lpips')
        iqa_dists = pyiqa.create_metric('dists')
        iqa_niqe = pyiqa.create_metric('niqe')
    net.eval()
    if net_Sup:
        net_Sup.eval()
    eval_results = {'eval_input_psnr': 0.0, 'eval_output_psnr': 0.0, 
                    'eval_input_ssim': 0.0, 'eval_output_ssim': 0.0, 
                    'eval_input_lpips': 0.0, 'eval_output_lpips': 0.0,
                    'eval_input_dists': 0.0, 'eval_output_dists': 0.0,
                    'eval_input_niqe': 0.0, 'eval_output_niqe': 0.0,  
                    'infer_time': 0.0}
    with torch.no_grad():
        for data_in, label, name in eval_loader:
            inputs, labels = Variable(data_in).to(DEVICE), Variable(label).to(DEVICE)
            start_time = time.time()
            if args.experiment_name == 'ERRNet':
                _, C, H, W = inputs.shape
                hypercolumn = net_Sup(inputs)
                hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
                inputs_with_hypercolumn = torch.cat([inputs] + hypercolumn, dim=1)
                outputs = net(inputs_with_hypercolumn)
            else:
                outputs, *_ = net(inputs) if not net_Sup else net(inputs, net_Sup(inputs))
            # Ensure outputs have the correct number of dimensions
            if outputs.dim() < 4:
                outputs = outputs.unsqueeze(0)  # Add a batch dimension if missing
            eval_results['infer_time'] += time.time() - start_time
            eval_results['eval_input_psnr'] += iqa_psnr(inputs, labels).item()
            eval_results['eval_output_psnr'] += iqa_psnr(torch.clamp(outputs, 0., 1.), labels).item()
            eval_results['eval_input_ssim'] += iqa_ssim(inputs, labels).item()
            eval_results['eval_output_ssim'] += iqa_ssim(torch.clamp(outputs, 0., 1.), labels).item()
            eval_results['eval_input_lpips'] += iqa_lpips(inputs, labels).item()
            eval_results['eval_output_lpips'] += iqa_lpips(torch.clamp(outputs, 0., 1.), labels).item()
            eval_results['eval_input_dists'] += iqa_dists(inputs, labels).item()
            eval_results['eval_output_dists'] += iqa_dists(torch.clamp(outputs, 0., 1.), labels).item()
            eval_results['eval_input_niqe'] += iqa_niqe(inputs).item()
            eval_results['eval_output_niqe'] += iqa_niqe(torch.clamp(outputs, 0., 1.)).item()

            if save_results:
                save_path = os.path.join(args.save_folder, args.strategy, args.experiment_name + '_test_results', dataset_name)
                os.makedirs(save_path, exist_ok=True)
                output_image = np.squeeze(torch.clamp(outputs, 0., 1.).cpu().numpy()).transpose((1, 2, 0))
                img.imsave(os.path.join(save_path, f"{name[0].split('.')[0]}.png"), np.uint8(output_image * 255.))
        
        eval_results['eval_input_psnr'] /= len(eval_loader)
        eval_results['eval_output_psnr'] /= len(eval_loader)
        eval_results['eval_input_ssim'] /= len(eval_loader)
        eval_results['eval_output_ssim'] /= len(eval_loader)
        eval_results['eval_input_lpips'] /= len(eval_loader)
        eval_results['eval_output_lpips'] /= len(eval_loader)
        eval_results['eval_input_dists'] /= len(eval_loader)
        eval_results['eval_output_dists'] /= len(eval_loader)
        eval_results['eval_input_niqe'] /= len(eval_loader)
        eval_results['eval_output_niqe'] /= len(eval_loader)

        eval_results['eval_input_psnr'] = round(eval_results['eval_input_psnr'], 2)
        eval_results['eval_output_psnr'] = round(eval_results['eval_output_psnr'], 2)
        eval_results['eval_input_ssim'] = round(eval_results['eval_input_ssim'], 3)
        eval_results['eval_output_ssim'] = round(eval_results['eval_output_ssim'], 3)
        eval_results['eval_input_lpips'] = round(eval_results['eval_input_lpips'], 3)
        eval_results['eval_output_lpips'] = round(eval_results['eval_output_lpips'], 3)
        eval_results['eval_input_dists'] = round(eval_results['eval_input_dists'], 3)
        eval_results['eval_output_dists'] = round(eval_results['eval_output_dists'], 3)
        eval_results['eval_input_niqe'] = round(eval_results['eval_input_niqe'], 3)
        eval_results['eval_output_niqe'] = round(eval_results['eval_output_niqe'], 3)

    return eval_results

if __name__ == '__main__':
    import warnings
    # Ignore all warnings
    warnings.filterwarnings("ignore")

    args = parse_arguments()
    logging.info(f"Experiment Name: {args.experiment_name}")
    logging.info(f"Strategy: {args.strategy}")
    net, net_Sup = load_model(args)
    args.eval_paths = [
        'Nature20, ../TestDatasets/reflection-removal/test/Nature/blended/, ../TestDatasets/reflection-removal/test/Nature/transmission_layer/', 
        'Real20, ../TestDatasets/reflection-removal/test/real20_420/blended/, ../TestDatasets/reflection-removal/test/real20_420/transmission_layer/',
        'SIR2, ../TestDatasets/reflection-removal/test/SIR2/blended/, ../TestDatasets/reflection-removal/test/SIR2/transmission_layer/',
        'OpenRR-1k (val), ../TestDatasets/clean_NTIRE2025_Challenge/val_100/blended/, ../TestDatasets/clean_NTIRE2025_Challenge/val_100/transmission_layer/',
        'OpenRR-1k (test), ../TestDatasets/clean_NTIRE2025_Challenge/test_100/blended/, ../TestDatasets/clean_NTIRE2025_Challenge/test_100/transmission_layer/'] 
    datasets = [tuple(path.split(', ')) for path in args.eval_paths]
    for dataset_name, in_path, gt_path in datasets:
        eval_loader = get_eval_loader(in_path, gt_path)
        results = test_model(net, net_Sup, eval_loader, dataset_name, args.SAVE_test_Results)
        logging.info(f"Results for {dataset_name}: {results}")