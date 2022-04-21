import numpy as np
import os, random
import pickle
import PIL
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from keras.utils import Sequence
# For Two Encoder Net
from keras.models import load_model
from keras import backend as K

# Set seeds for RNGs
def random_init(seed):
	random.seed(seed)
	np.random.seed(seed)

# Image/file I/O
def get_filenames(directory):
	filenames = [f for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]
	random.shuffle(filenames)
	return filenames

def save_image(image, directory, filename):
	filename = os.path.join(directory, filename)
	save_img(filename, image)

def save_file(obj, directory, filename):
	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = os.path.join(directory, filename)
	with open(filename, "wb") as ofile:
		pickle.dump(obj, ofile)

# Image processing utils
def random_crop(image, width, height):
	row = np.random.randint(image.shape[0]-height+1)
	col = np.random.randint(image.shape[1]-width+1)
	return image[row:row+height, col:col+width, ...]

def image_resize(image, size):
	return img_to_array(array_to_img(image).resize(size, resample=PIL.Image.LANCZOS))/255.0

def add_QIS_noise(image, alpha, read_noise, nbits=3):
	pix_max = 2**nbits-1
	frame = np.random.poisson(alpha*image) + read_noise*np.random.randn(*image.shape)
	frame = np.round(frame)
	frame = np.clip(frame, 0, pix_max)
	noisy = frame.astype(np.float32) / alpha
	return noisy


def load_crops(directory, filenames, patch_sz, num_patch, jit=2, J=2):
	size = num_patch*len(filenames)
	window_sz = (patch_sz + 2*jit) * J
	crops = np.empty([size, window_sz, window_sz, 1])
	cnt = 0
	for fname in filenames:
		image = img_to_array(load_img(os.path.join(directory, fname), color_mode="grayscale"))/255.0
		[height, width, _]	= image.shape
		if height < window_sz or width < window_sz:
			continue
		for i in range(num_patch):
			crops[cnt,:,:,:] = random_crop(image, window_sz, window_sz)
			cnt += 1
	crops = crops[:cnt]
	print("Loaded %d instances"%cnt)
	return crops, cnt

def make_burst(image, burst_sz, patch_sz, jit, J, rd_crop=False):
	stack = []
	static_stack = []
	# Decide burst direction
	if not rd_crop:
		x1, x2 = np.random.randint(2*jit*J+1, size=2)
		xs = np.linspace(x1, x2, num=burst_sz)
		ys = np.linspace(0, 2*jit*J, num=burst_sz)
		if np.random.random() < 0.5:
			xs, ys = ys, xs
	# Generate frames
	for i in range(burst_sz):
		frame = image
		if rd_crop:
			if i == 0:
				frame = frame[jit*J:(jit+patch_sz)*J, jit*J:(jit+patch_sz)*J, ...] # center
			else:
				frame = random_crop(frame, patch_sz*J, patch_sz*J)
		else:
			row, col = int(round(xs[i])), int(round(ys[i]))
			frame = frame[row:row+patch_sz*J, col:col+patch_sz*J, ...]
		frame = image_resize(frame, (patch_sz, patch_sz))
		stack.append(frame[:,:,0])
	return np.transpose(stack, [1,2,0]) # channel last

def gen_data(crops, patch_sz, burst_sz, jit=2, J=2, rd_crop=False):
	cnt = len(crops)
	y = np.empty([cnt, patch_sz, patch_sz, 1]) # ground truth
	x = np.empty([cnt, patch_sz, patch_sz, burst_sz]) # noisy burst
	for i in range(cnt):
		stack = make_burst(crops[i], burst_sz, patch_sz, jit, J, rd_crop=rd_crop)
		y[i,:,:,0] = stack[:,:,0] if rd_crop else stack[:,:,burst_sz//2]
		x[i,:,:,:] = stack
	return y, x, cnt






# Training data generator for Two Encoder Net that generates different noisy samples in each epoch
class BurstSequence2E(Sequence):
	def __init__(self, directory, patch_sz, num_patch, burst_sz, batch_sz, 
			jit=2, J=2, rd_crop=False, noise=True, alpha=4.0, read_noise=0.25, 
			regen_after=0, renoi_after=0, is_train=False, shift_ckpt=None, noisy_ckpt=None):
		# Store crops and generate data
		filenames = get_filenames(directory)
		self.crops, self.num_burst = load_crops(directory, filenames, patch_sz, num_patch, jit=jit, J=J)
		self.y, self.xc, _ = gen_data(self.crops, patch_sz, burst_sz, jit=jit, J=J, rd_crop=rd_crop)
		self.x = add_QIS_noise(self.xc, alpha, read_noise) if noise else self.xc
		# Store param for data generating
		self.patch_sz = patch_sz
		self.burst_sz = burst_sz
		self.jit = jit
		self.J = J
		self.rd_crop = rd_crop
		# Store other param
		self.batch_sz = batch_sz
		self.noise = noise
		self.alpha = alpha
		self.read_noise = read_noise
		self.regen_after = regen_after
		self.renoi_after = renoi_after
		self.is_train = is_train
		# Epoch counter
		self.curr_epoch = 0


		# =================================================================================
		# Note: Please define here your own function to extract features from teachers.
		#
		# The load_functor() function serves as a general feature loader, which returns the
		# features of the n-th layer. 
		# =================================================================================
		def dummy(y_true, y_pred):
			return K.sum(y_pred)
		def load_functor(model, nlayer):
			inputs = model.input # input placeholder
			outputs = [layer.output for layer in model.layers][1:nlayer+1] # 1st layer to end of encoder
			return K.function([inputs, K.learning_phase()], outputs)  # evaluation function
		if is_train:
			model = load_model(noisy_ckpt, custom_objects={'l2_loss': dummy, 'psnr_metric': dummy})
			self.noisy_enc = load_functor(model, 19) # Extract features from the 19-th layer
			model = load_model(shift_ckpt, custom_objects={'l2_loss': dummy, 'annealed_loss': dummy, 'encoder_loss': dummy, 'psnr_metric': dummy})
			self.shift_enc = load_functor(model, 19) # Extract features from the 19-th layer
		else:
			self.noisy_enc, self.shift_enc = None, None
		

	def __len__(self):
		return int(np.ceil(self.num_burst / float(self.batch_sz)))

	def __getitem__(self, idx):
		y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
		x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
		x_shift = self.xc[idx * self.batch_sz:(idx + 1) * self.batch_sz]
		# Generate "correct" codes from pretrained encoders
		if self.is_train:
			x_static = np.repeat(y, self.burst_sz, axis=-1)
			x_noisy = add_QIS_noise(x_static, self.alpha, self.read_noise)
			c1_true = self.noisy_enc([x_noisy, 1.])[-1]
			c2_true = self.shift_enc([x_shift, 1.])[-1]
		else:
			c1_true = np.zeros((self.batch_sz, 64, 64, 64))
			c2_true = np.zeros((self.batch_sz, 64, 64, 64))
		return x, {"xNet1": c1_true, "xNet2": c2_true, "outputs": y}

	# Regenerate noise and burst after several epochs
	def on_epoch_end(self):
		self.curr_epoch += 1
		if self.jit > 0 and self.regen_after > 0 and self.curr_epoch % self.regen_after == 0:
			self.y, self.xc, _ = gen_data(self.crops, self.patch_sz, self.burst_sz, jit=self.jit, J=self.J, rd_crop=self.rd_crop)
			if not (self.noise and self.renoi_after > 0):
				self.x = self.xc
		if self.noise and self.renoi_after > 0 and self.curr_epoch % self.renoi_after == 0:
			self.x = add_QIS_noise(self.xc, self.alpha, self.read_noise)

	# Get noisy, shift, burst, and ground truth for presentation
	def get_images(self, idx):
		y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
		x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
		x_shift = self.xc[idx * self.batch_sz:(idx + 1) * self.batch_sz]
		x_static = np.repeat(y, self.burst_sz, axis=-1)
		x_noisy = add_QIS_noise(x_static, self.alpha, self.read_noise)
		images = {"x_shift": x_shift, "x_noisy": x_noisy, "x_burst": x, "y": y}
		return images


