'''
train with better visualization
'''
import argparse, os

from model.utils import *
from model.REDNet import *
import torch.optim as optim
import time
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

parser	= argparse.ArgumentParser(description='')

parser.add_argument('--patches',	dest='num_patch',	type=int,	default=8,			help='number of patches extracted from an image')
parser.add_argument('--patchsize',	dest='patch_sz',	type=int,	default=128,		help='patch size, height/width of frames in each burst')
parser.add_argument('--burstsize',	dest='burst_sz',	type=int,	default=8,			help='burst size, number of frames in each burst')
parser.add_argument('--batchsize',	dest='batch_sz',	type=int,	default=4,			help='batch size, number of bursts in each batch')

parser.add_argument('--epochs',		dest='epochs',		type=int,	default=10,		help='number of epoch')
parser.add_argument('--lr',			dest='lr_init',			type=float,	default=0.001,		help='learning rate')

parser.add_argument('--jit',		dest='jit',			type=int,	default=0,			help='jit')
parser.add_argument('--J',			dest='J',			type=int,	default=2,			help='downsample ratio J')
parser.add_argument('--alpha',		dest='alpha',		type=float,	default=4.0,		help='alpha')
parser.add_argument('--read_noise',		dest='read_noise',		type=float,	default=0.25,		help='read noise')

parser.add_argument('--nbits',		dest='nbits',		type=int,	default=3,		help='numbef of bit for QIS')
parser.add_argument('--train_path',	dest='train_path',				default='./datasample/VOC2012/train_data',				help='the directory that training images are saved')
parser.add_argument('--test_path',	dest='test_path',				default='./datasample/VOC2012/test_data',				help='the directory that validation images are saved')

parser.add_argument('--directory',	dest='directory',				default='./experiment',				help='the directory to save experiment results')
parser.add_argument('--model_name',	dest='model_name',				default='Rednet',				help='name of the experiment model')
parser.add_argument('--ckpt_pth',	dest='ckpt_pth',				default='',								help='the file of ckpt for resume')

parser.add_argument('--seed',		dest='seed',		type=int,	default=42,			help='random seed number')
args	= parser.parse_args([])


# seed = 42
# train_path = '/Users/zhangyunping/Library/CloudStorage/OneDrive-connect.hku.hk/PhD/ECCV2020_Dynamic-master/datasample/VOC2012/train_data'
# test_path = '/Users/zhangyunping/Library/CloudStorage/OneDrive-connect.hku.hk/PhD/ECCV2020_Dynamic-master/datasample/VOC2012/test_data'
# directory = '/Users/zhangyunping/Library/CloudStorage/OneDrive-connect.hku.hk/PhD/ECCV2020_Dynamic-master/experiment'
# model_name = 'Rednet'
# ckpt_pth = ''

# patch_sz = 128
# num_patch = 8
# burst_sz = 8
# batch_sz = 4
# alpha = 4
# read_noise = 0.25
# jit = 0
# J = 2
noise = True
channel_first = True
rd_move = False
# nbits = 3
binning = True
adam = True
# lr_init = 0.001
# epochs = 10
resume = False


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s",level=logging.INFO)

def train_epoch(model,dataloader, optimizer, epoch_idx):
    model.train()
    # for k,v in model.named_parameters():
    #     v.requires_grad =True
    pbar = enumerate(dataloader)
    nb = len(dataloader)

    pbar = tqdm(pbar,total = nb)
    epoch_loss = torch.zeros(1, device=model.device)
    avg_psnr = torch.zeros(1, device=model.device)
    logger.info(('\n' + '%10s' * 4)%('Epoch', 'memory', 'l2-loss','PSNR'))
    optimizer.zero_grad()

    for batch_i, (x,y) in pbar:
        x = x.to(model.device, non_blocking=True)
        y = y.to(model.device)
        output = model(x)
        loss = l2_loss(y_pred=output, y_true=y)
        batch_psnr, psnr = psnr_metric(y_pred=output,y_true=y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss
        avg_psnr += psnr
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        info = ('%10s'*2 + '%10.4g' * 2)%(
            '%g' % (epoch_idx), mem, epoch_loss.detach().numpy()/(batch_i+1), avg_psnr.detach().numpy()/(batch_i+1)
        )
        pbar.set_description(info)
        break

    return epoch_loss,avg_psnr

def eval_epoch(model,dataloader):
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=len(dataloader))
    epoch_loss = torch.zeros(1, device=model.device)
    avg_psnr = torch.zeros(1, device=model.device)


    logger.info('\n Evaluation ===============================================')
    logger.info(('\n'+'%10s' * 2)%('l2-loss','PSNR'))
    model.eval()
    with torch.no_grad():
        for batch_i, (x,y) in pbar:
            x = x.to(model.device, non_blocking=True)
            y = y.to(model.device)
            output = model(x)
            loss = l2_loss(y_pred=output, y_true=y)
            _, psnr = psnr_metric(y_pred=output,y_true=y)
            epoch_loss += loss
            avg_psnr += psnr
            info = ('%10.4g' * 2) % (loss.detach().numpy()/(batch_i+1), avg_psnr.detach().numpy()/(batch_i+1))
            pbar.set_description(info)
            break

    return epoch_loss,avg_psnr



if __name__=="__main__":

    logger.info("\n Initialization the params")
    random_init(seed=args.seed)
    save_dir = os.path.join(args.directory,args.model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    tb_writer = SummaryWriter(save_dir)
    last_pth = os.path.join(save_dir,'last.pt')
    best_pth = os.path.join(save_dir,'best.pt')
    best_psnr = 0



    logger.info("\n Loading the training data...")


    dataset_train = BurstData(directory =args.train_path, patch_sz=args.patch_sz,batch_sz=args.batch_sz,num_patch=args.num_patch,burst_sz=args.burst_sz,alpha=args.alpha,
                              read_noise=args.read_noise, jit=args.jit, J=args.J, channel_first=channel_first,noise=noise,rd_move=rd_move,
                              is_train=True,nbits=args.nbits,binning=binning)
    dataset_test = BurstData(directory =args.test_path, patch_sz=args.patch_sz,batch_sz=args.batch_sz,num_patch=args.num_patch,burst_sz=args.burst_sz,alpha=args.alpha,
                             read_noise=args.read_noise, jit=args.jit, J=args.J, channel_first=channel_first,noise=noise,rd_move=rd_move,
                             is_train=True,nbits=args.nbits,binning=binning)
    logger.info("\n Dataset loaded. Train:{:d} Test:{:d}".format(len(dataset_train),len(dataset_test)))
    train_loader = DataLoader(dataset_train, batch_size=5,collate_fn=dataset_train.collect_fn)
    test_loader = DataLoader(dataset_test, batch_size=5,collate_fn=dataset_test.collect_fn)

    # Build model
    logger.info("\n Building model...")
    model = REDNet15(num_blocks=15,num_features=64,input_channel=1,output_channel=1)
    logger.info(model)
    if torch.cuda.is_available():
        model = model.to('cuda')
        model = torch.nn.DataParallel(model).cuda()
        model.device = torch.device('cuda')
    else:
        model = torch.nn.DataParallel(model)
        model.device = torch.device('cpu')
    # define optimizer
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init)

    # check whether restore
    if resume:
        ckpt = torch.load(args.ckpt_pth)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['last_epoch']
        loss = ckpt['loss']
        model_name = ckpt['model_name']
        logger.info("\n successfully load model %10s" % (model_name))
    else:
        model_name = args.model_name
        start_epoch = 0

    end_epoch = start_epoch + args.epochs


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience =3, verbose=True)
    scheduler.last_epoch = start_epoch - 1

    # Start training
    logger.info('\n Start training process=======================')
    for epo_idx in range(start_epoch,end_epoch,1):
        train_loss,train_psnr = train_epoch(model,train_loader,optimizer,epo_idx)
        eval_loss,eval_psnr = eval_epoch(model,test_loader)

        # write log
        current_lr = optimizer.param_groups[0]['lr']
        tags = ['learning_rate','train/loss','train/psnr','val/loss','val/psnr']
        for x, tag in zip([np.array(current_lr),train_loss.detach().numpy(),train_psnr.detach().numpy()
                              ,eval_loss.detach().numpy(),eval_psnr.detach().numpy()], tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epo_idx)

        # save the last
        ckpt = {
            'model_state_dict': model.state_dict(),
            'last_epoch': epo_idx,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': model_name,
            'loss': train_loss.detach().numpy(),
        }
        torch.save(ckpt, last_pth)
        logger.info("Epoch {:d} saved".format(epo_idx))

        # update the best
        if eval_psnr > best_psnr:
            best_psnr = eval_psnr
        if best_psnr == eval_psnr:
            torch.save(ckpt, best_pth)
            logger.info("Epoch {:d} is the best currently".format(epo_idx))

        # generate the visualization images
        model.eval()
        vis_path = os.path.join(save_dir,'visulization_result')
        if not os.path.isdir(vis_path):
            os.makedirs(vis_path)
        name_input = os.path.join(vis_path,'Epoch{:d}_test_input.png'.format(epo_idx))
        name_output =  os.path.join(vis_path,'Epoch{:d}_test_output.png'.format(epo_idx))
        name_gt =  os.path.join(vis_path,'Epoch{:d}_test_gt.png'.format(epo_idx))
        with torch.no_grad():
            images = dataset_test.get_images(idx=1)
            x_binary = images['x_binary']
            x_noisy = images['x_noisy']
            cln_img = images['y']

            noisy_img = x_noisy[0,:,:,0]*255
            noisy_img = Image.fromarray(noisy_img.astype(np.uint8))
            noisy_img.save(name_input)

            gt_img = cln_img[0,:,:,0]*255
            gt_img = Image.fromarray(gt_img.astype(np.uint8))
            gt_img.save(name_gt)

            x_bined_noise = images['x_binned_noisy']
            x_bined_noise = torch.tensor(x_bined_noise[0,:,:,:]).to(model.device).unsqueeze(0)
            x_bined_noise = x_bined_noise.permute([0,3,1,2])
            y_pred = model(x_bined_noise)
            y = torch.tensor(cln_img[0,:,:,:]).to(model.device).unsqueeze(0)
            loss = l2_loss(y_pred=y_pred, y_true=y)
            _, psnr = psnr_metric(y_pred=y_pred,y_true=y)

            y_pred = y_pred.detach().numpy()
            y_pred_img = y_pred[0,0,:,:,]*255
            y_pred_img = Image.fromarray(y_pred_img.astype(np.uint8))
            y_pred_img.save(name_output)
            info = ('%10.4g' * 2) % (loss.detach().numpy(), psnr.detach().numpy())
            msg = ('For visualization, the loss is {:10.4f} and the psnr is {:10.4f} at epoch {:d}'
                   ).format(loss.detach().numpy(), psnr.detach().numpy(),epo_idx)
            logger.info(msg)






