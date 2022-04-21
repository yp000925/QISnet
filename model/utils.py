import numpy as np
import torch
import PIL
from PIL import Image,ImageOps
import os,random
from torch.utils.data import Dataset,DataLoader

def random_init(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_filenames(directory):
    filenames = [f for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]
    random.shuffle(filenames)
    return filenames

# generate the patches from each image randomly
def load_crops(directory, filenames, patch_sz, num_patch, jit=2, J=2):
    '''
    generate the randomly cropped patches for each clear image in the directory
    then stack them together

    :param directory:
    :param filenames:
    :param patch_sz:
    :param num_patch: how many patches is in each image
    :param jit:
    :param J: downsampling ratio
    :return: random cropped image with size (patch_sz+2*jit)*J
    '''
    total_num = num_patch*len(filenames)
    window_sz = (patch_sz+2*jit)*J
    crops = np.empty([total_num, window_sz, window_sz,1])
    cnt = 0
    for fname in filenames:
        image = np.array(Image.open(os.path.join(directory,fname)).convert('L'))/255.0
        [height, width] = image.shape
        if height<window_sz or width<window_sz:
            continue
        for i in range(num_patch):
            crops[cnt,:,:,0] = random_crop(image,window_sz,window_sz)
            cnt+=1
    crops = crops[:cnt]
    print("Loaded %d instances"%cnt)
    print("shape is ",crops.shape)
    return crops, cnt


# Image processing utils
def random_crop(image, width, height):
    row = np.random.randint(image.shape[0]-height+1)
    col = np.random.randint(image.shape[1]-width+1)
    return image[row:row+height, col:col+width]

def make_burst(image, burst_sz, patch_sz, jit, J, rd_move=False):
    stack = []
    static_stack = []
    # Decide burst direction
    # this is for continuous movement across the consecutive frames
    if jit == 0:
        frame = image_resize(image[:,:,0], (patch_sz,patch_sz))
        stack = [frame]*burst_sz
        # stach =  np.repeat(frame, burst_sz, axis =-1)
    else:
        if not rd_move:
            x1,x2 = np.random.randint(2*jit*J+1,size=2)
            xs = np.linspace(x1,x2,num=burst_sz)
            ys = np.linspace(0,2*jit*J, num=burst_sz)
            if np.random.random()<0.5:
                xs,ys = ys,xs

        #Generate frames
        for i in range(burst_sz):
            frame = image
            if rd_move:
                if i == 0:
                    frame = frame[jit*J:(jit+patch_sz)*J, jit*J:(jit+patch_sz)*J, ...] # center
                else:
                    frame = random_crop(frame,patch_sz*J,patch_sz*J)
            else:
                row, col = int(round(xs[i])),int(round(ys[i]))
                frame = frame[row:row+patch_sz*J, col:col+patch_sz*J, ...]
            frame = image_resize(frame[:,:,0], (patch_sz,patch_sz))
            stack.append(frame)
    return np.transpose(stack, [1,2,0])

def image_resize(img_arr,size):
    if img_arr.max()>1:
        print("Check the image array value here, should be normalized to 0-1")
        raise ValueError
    img_arr = img_arr*255
    img = Image.fromarray(img_arr.astype(np.uint8)).resize(size)
    return np.array(img)/255.0

def gen_data(crops, patch_sz, burst_sz, jit=0, J=2, rd_move=False):
    '''
    :param crops: cropped images
    :param patch_sz: patch dimension after downsampling
    :param burst_sz: number of burst frames in for each cropped patch
    :param jit: incidate the movement in pixel scale across burst time
    :param J: downsampling ratio
    :param rd_move: indicate the movement is linear or randomly -> if linear, the GT will be the middle point(frame)
    :return:
        y: GT with dimension [#cropped patches, patch_sz, patch_sz, 1]
        x: Burst observation with dimension [#cropped patches, patch_sz, patch_sz, burst_sz]
        cnt: #cropped_patches

    '''
    cnt = len(crops)
    y = np.empty([cnt,patch_sz,patch_sz,1]) #ground truth
    x = np.empty([cnt,patch_sz,patch_sz,burst_sz]) #burst images with noise
    for i in range(cnt):
        stack = make_burst(crops[i], burst_sz,patch_sz,jit,J,rd_move)
        y[i,:,:,0] = stack[:,:,0] if rd_move else stack[:,:,burst_sz//2] #取的是均匀movement里面的中间时刻(frame)的值作为GT
        x[i,:,:,:] = stack
    return y, x, cnt


def add_QIS_noise(image, alpha, read_noise, nbits=3):
    pix_max = 2**nbits-1
    frame = np.random.poisson(alpha*image) + read_noise*np.random.randn(*image.shape)
    frame = np.round(frame)
    frame = np.clip(frame, 0, pix_max)
    noisy = frame.astype(np.float32) / alpha
    return noisy

# generate training data from clean CMOS image (GT)
class BurstData(Dataset):
    def __init__(self, directory, patch_sz, num_patch, burst_sz,batch_sz, alpha=4, read_noise=0.25, jit=0, J=2, channel_first = True, noise = True,
                 rd_move = False, is_train=False,nbits=3,binning = True):
        # super(BurstData,self).__init__()
        # Generate randomly cropped data
        # Based on the cropped clean data, the synthetic burst QIS data is generated with the read noise disturbance
        self.imgs_name = get_filenames(directory)
        self.directory = directory
        # Initialize the gt and noisy images
        self.y = None
        self.x = None
        self.xc = None

        # Store params for data generation
        self.patch_sz = patch_sz
        self.burst_sz = burst_sz
        self.jit = jit
        self.J = J
        self.rd_move = rd_move
        self.num_patch = num_patch
        self.nbits = nbits

        # Store other param
        self.batch_sz = batch_sz
        self.noise = noise
        self.alpha = alpha
        self.read_noise = read_noise
        self.is_train = is_train
        self.channel_first = channel_first
        self.binning =binning

        #Epoch counter
        self.current_epoch = 0

    def __getitem__(self, idx):
        img_name = self.imgs_name[idx]
        crops = self.generate_crop(self.directory,img_name,self.patch_sz,self.num_patch,self.jit,self.J)
        self.y, self.xc, _ = self.gen_data(crops,self.patch_sz,self.burst_sz,self.jit, self.J, rd_move=self.rd_move,binning=self.binning)
        # print("Ground truth size is ", self.y.shape)
        # print("Burst obsearvation size is ", self.xc.shape)
        self.x = add_QIS_noise(self.xc,self.alpha,self.read_noise,self.nbits) if self.noise else self.xc
        return torch.from_numpy(self.x), torch.from_numpy(self.y)

    def get_images(self, idx):
        imgs_name = self.imgs_name[idx]
        crops= self.generate_crop(self.directory,imgs_name,self.patch_sz,self.num_patch,self.jit,self.J)
        y, xc, _ = self.gen_data(crops,self.patch_sz,self.burst_sz,self.jit, self.J, rd_move=self.rd_move, binning = False)
        # print("Ground truth size is ", self.y.shape)
        # print("Burst obsearvation size is ", self.xc.shape)
        x = add_QIS_noise(xc,self.alpha,self.read_noise,self.nbits)
        images = {"x_shift": xc, "x_noisy": x, "y": y}
        if self.binning:
            L = xc.shape[-1]
            xbin = np.sum(xc, axis=3,keepdims=True)
            xbin = 1-xbin/L
            xbin_noise = add_QIS_noise(xbin,self.alpha,self.read_noise,self.nbits)
            images = {"x_binary": xc, "x_noisy": x, "y": y, "x_binned_noisy":xbin_noise}
        return images

    def __len__(self):
        # return self.num_patch*len(self.imgs_name)
        return len(self.imgs_name)


    def generate_crop(self, directory, filename, patch_sz, num_patch, jit=2, J=2):
        '''
        generate the randomly cropped patches for each clear image in the directory
        then stack them together

        :param directory:
        :param filenames:
        :param patch_sz:
        :param num_patch: how many patches is in each image
        :param jit:
        :param J: downsampling ratio
        :return: random cropped image with size (patch_sz+2*jit)*J
        '''

        window_sz = (patch_sz+2*jit)*J
        crops = np.empty([num_patch, window_sz, window_sz,1])
        image = np.array(Image.open(os.path.join(directory,filename)).convert('L'))/255.0
        [height, width] = image.shape
        if height< window_sz or width < window_sz:
            print("the window_sz is larger than the image dimension")
            print("image dimension:",height,width)
            print("window_sz",window_sz)
            raise ValueError
        for i in range(num_patch):
            crops[i,:,:,0] = random_crop(image,window_sz,window_sz)
        # print("shape is ",crops.shape)
        return crops

    def gen_data(self, crops, patch_sz, burst_sz, jit=0, J=2, rd_move=False, binning= True):
        '''
        :param crops: cropped images
        :param patch_sz: patch dimension after downsampling
        :param burst_sz: number of burst frames in for each cropped patch
        :param jit: incidate the movement in pixel scale across burst time
        :param J: downsampling ratio
        :param rd_move: indicate the movement is linear or randomly -> if linear, the GT will be the middle point(frame)
        :return:
            y: GT with dimension [#cropped patches, patch_sz, patch_sz, 1]
            x: Burst observation with dimension [#cropped patches, patch_sz, patch_sz, burst_sz]
            cnt: #cropped_patches

        '''
        cnt = len(crops)
        y = np.empty([cnt,patch_sz,patch_sz,1]) #ground truth
        x = np.empty([cnt,patch_sz,patch_sz,burst_sz]) #burst images with noise
        for i in range(cnt):
            stack = make_burst(crops[i], burst_sz,patch_sz,jit,J,rd_move)
            y[i,:,:,0] = stack[:,:,0] if rd_move else stack[:,:,burst_sz//2] #取的是均匀movement里面的中间时刻(frame)的值作为GT
            x[i,:,:,:] = stack
        if binning:
            L = x.shape[-1]
            x = np.sum(x, axis=3, keepdims=True)
            x = x/L
        return y, x, cnt

    def collect_fn(self,data):
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        if self.channel_first:
            x = x.permute(0,3,1,2)
            y = y.permute(0,3,1,2)
        return x,y




if __name__=="__main__":
    # img = np.random.random((512,512,1))
    # img = img/img.max()
    patch_sz = 128
    J=2
    binning = True
    directory = '/home/zhangyp/PycharmProjects/QISnet/VOC2012/train_data'
    dataset = BurstData(directory,patch_sz=patch_sz,num_patch=4,burst_sz=2,batch_sz=16,alpha=4,read_noise=0.25,
                        jit=0,J=J,channel_first=True,noise=True,rd_move=False,is_train=True,nbits=3,binning=binning)
    dataloader = DataLoader(dataset, batch_size=5,collate_fn=dataset.collect_fn)
    # device = torch.device('cpu')
    for batch_i, (x,y) in enumerate(dataloader):
        continue
    #     images = dataset.get_images(idx=4)
    #     x_binary = images['x_binary']
    #     x_noisy = images['x_noisy']
    #     cln_img = images['y']
    #     noisy_img = x_noisy[0,:,:,0]*255
    #     noisy_img = Image.fromarray(noisy_img.astype(np.uint8))
    #     noisy_img.show()
    #     gt_img = cln_img[0,:,:,0]*255
    #     gt_img = Image.fromarray(gt_img.astype(np.uint8))
    #     gt_img.show()
    #     if binning:
    #         x_bined_noise = images['x_binned_noisy']
    #         x_bined_img = x_bined_noise[0,:,:,0]*255
    #         x_bined_img = Image.fromarray(x_bined_img.astype(np.uint8))
    #         x_bined_img.show()
    #     break
    #
    # for i in range(x_noisy.shape[-1]):
    #     img_data = x_noisy[0,:,:,i]*255
    #     img = Image.fromarray(img_data.astype(np.uint8))
    #     img.show()