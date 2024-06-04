import argparse
import lpips
import cv2
import os
import gc
import torch
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
	parser.add_argument('-a','--artist', type=str, default='Vincent_van_Gogh')
	parser.add_argument('-v','--version', type=str, default='0.1')
	parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
	return parser.parse_args()


def find_similar_images(file_path: str, compare_dir: str, use_gpu: bool = True, version: str = '0.1'):
    
	loss_fn = lpips.LPIPS(net='alex',version=version)
	if use_gpu:
		loss_fn.cuda()
    
	files = os.listdir(compare_dir)
	similar = []
	for file in tqdm(files, desc="Processing images"):
		compare_file_path = os.path.join(compare_dir, file)
		img1 = lpips.im2tensor(lpips.load_image(compare_file_path))
		img0 = lpips.im2tensor(cv2.resize(lpips.load_image(file_path),(img1.shape[3], img1.shape[2]))) # RGB image from [-1,1]

		if use_gpu:
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		with torch.no_grad():  # This ensures that no gradients are computed which can save memory
			dist = loss_fn.forward(img0, img1)
			dist = dist.item()
   
		if use_gpu:
			img0 = img0.cpu()
			img1 = img1.cpu()
			del img0
			del img1
			torch.cuda.empty_cache()
			gc.collect()

		# print('Distance: %.3f'%dist)
		similar.append([dist, file, os.path.basename(file_path)])
	return sorted(similar, key=lambda x: x[0])


def compute_similaty(file_path: str, compare_file_path: str, use_gpu: bool = True, version: str = '0.1'):
	loss_fn = lpips.LPIPS(net='alex',version=version)
	if use_gpu:
		loss_fn.cuda()

	img1 = lpips.im2tensor(lpips.load_image(compare_file_path))
	img0 = lpips.im2tensor(cv2.resize(lpips.load_image(file_path),(img1.shape[3], img1.shape[2]))) # RGB image from [-1,1]

	if use_gpu:
		img0 = img0.cuda()
		img1 = img1.cuda()

	# Compute distance
	with torch.no_grad():  # This ensures that no gradients are computed which can save memory
		dist = loss_fn.forward(img0, img1)
		dist = dist.item()

	if use_gpu:
		img0 = img0.cpu()
		img1 = img1.cpu()
		del img0
		del img1
		torch.cuda.empty_cache()
		gc.collect()

	print('Distance: %.3f'%dist)
	return dist

if __name__ == '__main__':
	opt = parse_args()
	BASELINE_DIR = f'data/images/style/baseline/{opt.artist}'
 
	similar = find_similar_images(opt.path0, BASELINE_DIR)
	similar = sorted(similar, key=lambda x: x[0])
	print(similar[:5])

	with open('similar.txt', 'w') as f:
		for item in similar:
			f.write(f'{item[0]}, {item[1]}\n')

	# python lpips_2imgs.py -p0 data/images/style/generated/Vincent\ van\ Gogh\ style.png -p1 data/images/style/baseline/Vincent_van_Gogh/Vincent_van_Gogh_10.jpg --use_gpu
	# python lpips_2imgs.py -p0 data/images/style/generated/Vincent\ van\ Gogh\ style.png -p1 data/images/style/baseline/Vincent_van_Gogh/Vincent_van_Gogh_368.jpg --use_gpu