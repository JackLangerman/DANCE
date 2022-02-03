from torch.utils.data import Dataset, DataLoader
import torch


import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
import natsort
import glob
from PIL import Image
from skimage.transform import resize
import random
import ioutils as io
import cv2
import heapq
import os
import sklearn.metrics as metrics
from tqdm import trange, tqdm
import math
import multiprocessing as mp
import pickle
import open3d as o3d
import itertools


from my_networks import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('./data/parameters.pic'):
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    K = np.asarray(io.load(K_path))
    pnts_array = np.asarray(pcd.points)
    # minimum coordinate and coordinate range for each axis (definded manually)
    ptp_min, ptp = np.array([-2.7, -8.6, -0.05]), np.array([5.8, 14.3, 3.05])
    mean_sc = ((pnts_array-ptp_min)/ptp).mean(0)
    mean_color = np.asarray(pcd.colors).mean(0)
    if not os.path.exists("./data"):
        os.mkdir("./data")
    dbfile = open('./data/parameters.pic', 'wb')      
    pickle.dump((ptp_min, ptp, mean_sc, mean_color, K), dbfile) 
    dbfile.close()
else:
    dbfile = open('./data/parameters.pic', 'rb')      
    ptp_min, ptp, mean_sc, mean_color, K = pickle.load(dbfile) 
    dbfile.close()
    

    
def error_plot(a,b=None,metric=metrics.pairwise.paired_distances, 
    percentiles=[50, 95], units='', mean=True, horiz=True, bins='auto',
    min_err_clip=-float('inf'), max_err_clip=float('inf')):
    '''
        args:
            a : either list of vectors or (iff b is None) list of errors 
            b : either paired list of vectors with a or None -- default=None
            metric : distance function to compare a and b -- default=euclidean
            percentiles : list of percentiles to report -- default=[50, 95]
            units : units to report error in -- default=''
    '''
    errs = a if b is None else metric(a,b)

    if horiz:
        plt.figure(figsize=(14,5))
    else:
        plt.figure(figsize=(10,10))

    for i, cum in enumerate((True, False), start=1):
        if horiz:
            plt.subplot(1,2,i)
        else:
            plt.subplot(2,1,i)

        p, bins, patches = plt.hist(np.clip(errs, min_err_clip, max_err_clip), 
                                    bins=bins, histtype='step',
                                    cumulative=cum, density=True)

        for ptile in percentiles:
            ptile_val = np.percentile(errs, ptile)
            plt.plot([ptile_val]*2, (0,p.max()), 
                     label='{}th % = {:.2f} {}'.format(ptile, ptile_val, units))

        if mean:
            mu = np.clip(errs, min_err_clip, max_err_clip).mean()
            plt.plot((mu,)*2, (0,p.max()), label='mean = {:.2f} {}'.format(mu,units), linestyle='dashed')
            
        plt.title('cdf' if cum else 'pdf')
        plt.legend()
        plt.xlabel('error'+((' (%s)' % units) if units!='' else ''))
        plt.ylabel('P')
        plt.tight_layout()
    return errs

def compute_angle(r_predict, gt_R):
    r_err = np.matmul(r_predict, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    return r_err

def compute_translation(t_predict, gt_T):
    return np.sqrt(((t_predict-gt_T)**2).sum())

def compute_pose(scr, xofs, yofs, K, ptp_min, ptp, background_color, 
    PnPSolver, bg_eps, pnp_method, pnp_kwargs):
    
    scr2 = np.asarray(scr)
    mask = ~(np.linalg.norm(scr2 - background_color, axis=-1) < bg_eps)

    height, width = scr2.shape[:2]
    sc_points = scr2[mask]
    px = (
            np.mgrid[0:height,0:width].transpose(1,2,0)[mask] + 
            np.asarray([yofs, xofs])
        ).T

    objectPoints = sc_points.astype(np.float64) * ptp + ptp_min
    imagePoints = px.astype(np.float64)[(1,0), :].T
    cameraMatrix = K.astype(np.float64)
    distCoeffs = None
    #cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP
    res = PnPSolver(objectPoints, imagePoints, cameraMatrix, distCoeffs=distCoeffs, flags=pnp_method, **pnp_kwargs)

    if len(res)==4:
        ret, rvecs_hat, tvecs_hat, inliers = res
    elif len(res)==3:
        ret, rvecs_hat, tvecs_hat, inliers = *res, None
    T_hat = np.eye(4)
    
    T_hat[:3, :3], T_hat[:3, 3] = cv2.Rodrigues(rvecs_hat)[0], tvecs_hat.T
        
    T_hat[:3, :3], T_hat[:3, 3] = T_hat[:3, :3].T, -T_hat[:3, :3].T@T_hat[:3, 3]

    return T_hat, mask, (inliers, imagePoints)



def get_test_paths(test_img_path, test_scr_path, test_pose_path):
    pose_paths = natsort.natsorted(glob.glob(test_pose_path+'/**/*vecs.json', recursive=True))
    r_vecs = []
    t_vecs = []
    for i_pose in range(0, len(pose_paths), 2):
        assert pose_paths[i_pose].split('/')[-2] == pose_paths[i_pose+1].split('/')[-2], 'not the same folders'
        r_vecs.extend(io.load(pose_paths[i_pose]))
        t_vecs.extend(io.load(pose_paths[i_pose+1]))
    true_poses = []
    for i in range(len(r_vecs)):
        cur_pose = np.eye(4)
        cur_pose[:3, :3] = cv2.Rodrigues(np.squeeze(np.array(r_vecs[i])))[0]
        cur_pose[:3, 3] = np.array(t_vecs[i]).T
        true_poses.append(cur_pose[np.newaxis,:,:])
    true_poses = np.concatenate(true_poses)
    img_paths = natsort.natsorted(glob.glob(test_img_path+'/**/*.png', recursive=True))
    scr_label_paths = natsort.natsorted(glob.glob(test_scr_path+'/**/*.tiff', recursive=True))
    return img_paths, scr_label_paths, true_poses

def load_net(net_path, device):
    net = scr_net()
    state_dict = torch.load(net_path)
    try:
        net.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.','')
                new_state_dict[k]=v
        net.load_state_dict(new_state_dict)
    net.to(device)
    net.eval()
    return net

def test(net, dataloader, device):
    net.eval()
    scr_predicts, true_poses = [], []
    scr_labels = []
    input_imgs = []
    gen_imgs = []

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        inputs, scrs, poses = sample_batched['image'], sample_batched['scr'], sample_batched['pose']
        inputs = inputs.to(device, dtype=torch.float)
        
        with torch.no_grad():
            scr_predict = net(inputs)[1]
        
        input_imgs.append(inputs.cpu().numpy())
        scr_predicts.append(scr_predict.cpu().numpy())
        scr_labels.append(torch.nn.functional.interpolate(scrs.cpu(), scale_factor=1/8.0, mode='bilinear').numpy())
        true_poses.append(poses.numpy())
        
    input_imgs = np.concatenate(input_imgs)
    scr_predicts = np.concatenate(scr_predicts)
    scr_labels = np.concatenate(scr_labels)
    true_poses = np.concatenate(true_poses)
    print(scr_predicts.shape, scr_labels.shape, true_poses.shape)
    plt.imshow(scr_predicts[0,...].transpose(1,2,0))
    plt.show()
    
    return input_imgs, scr_predicts, scr_labels, true_poses


def make_render_dataset(main_path, train_flag=True):
    if train_flag:
        img_paths = natsort.natsorted(glob.glob(main_path+'/rendered/**/*.png', recursive=True))
        scr_paths = natsort.natsorted(glob.glob(main_path+'/scene_coords/**/*.tiff', recursive=True))
        pose_paths = natsort.natsorted(glob.glob(main_path+'/poses/**/*vecs.json', recursive=True))
    else:
        img_paths = natsort.natsorted(glob.glob(main_path+'/val_real_img/**/*.png', recursive=True))
        scr_paths = natsort.natsorted(glob.glob(main_path+'/val_scene_coords/**/*.tiff', recursive=True))
        pose_paths = natsort.natsorted(glob.glob(main_path+'/val_poses/**/*vecs.json', recursive=True))
    r_vecs = []
    t_vecs = []
    for i_pose in range(0, len(pose_paths), 2):
        assert pose_paths[i_pose].split('/')[-2] == pose_paths[i_pose+1].split('/')[-2], 'not the same folders'
        r_vecs.extend(io.load(pose_paths[i_pose]))
        t_vecs.extend(io.load(pose_paths[i_pose+1]))
        
    true_poses = []
    for i in range(len(r_vecs)):
        cur_pose = np.eye(4)
        cur_pose[:3, :3] = cv2.Rodrigues(np.squeeze(np.array(r_vecs[i])))[0]
        cur_pose[:3, 3] = np.array(t_vecs[i]).T
        true_poses.append(cur_pose[np.newaxis,:,:])
    true_poses = np.concatenate(true_poses)
    
    return img_paths, scr_paths, true_poses

class dataset_scr(Dataset):
    def __init__(self, img_paths, poses, scr_paths=None, transform=None, max_iters=None):
        self.img_paths = img_paths
        self.img_num = len(self.img_paths)
        if max_iters == None:
            self.dataset_len = self.img_num
        else:
            self.dataset_len = self.img_num * int(np.ceil(float(max_iters) / self.img_num))

        if scr_paths != None:
            self.scr_paths = scr_paths
        else:
            self.scr_paths = None
            
        self.poses = poses
        
        self.transform = transform
        
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        index = index % self.img_num
        img = np.asarray(Image.open(self.img_paths[index]).convert('RGB'))/255.0
        img = img.transpose(2, 0, 1)
        
        if self.scr_paths != None:
            scr = np.asarray(resize(skimage.io.imread(self.scr_paths[index]), (360,640)))
            scr = scr.transpose(2, 0, 1)
            
        pose = self.poses[index, ...]
                        
        if self.scr_paths != None:
            sample = {'image': img, 'pose': pose, 'scr': scr}
        else:
            sample = {'image': img, 'pose': pose}
            
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class random_crop(object):
    def __init__(self, validROI_x=16, validROI_y=5, validROI_w=594, validROI_h=341, crop_size=320):
        self.validROI_x = validROI_x
        self.validROI_y = validROI_y
        self.validROI_w = validROI_w
        self.validROI_h = validROI_h
        self.crop_size = crop_size
    def __call__(self, sample):
        w = self.validROI_w
        h = self.validROI_h

        crop_size_w, crop_size_h = self.crop_size, self.crop_size
        xoffs, yoffs = (self.validROI_x + np.random.randint(0, w-crop_size_w),
                        self.validROI_y + np.random.randint(0, h-crop_size_h))
        img, pose, scr = sample['image'], sample['pose'], sample['scr']
        return {'image': img[:,yoffs:yoffs+crop_size_h,xoffs:xoffs+crop_size_w], 
                'scr': scr[:,yoffs:yoffs+crop_size_h,xoffs:xoffs+crop_size_w],
                'pose': pose}
        
class normalize(object):
    def __init__(self, mean=0.5, divider=0.5):
        self.mean = mean
        self.divider = divider
    
    def __call__(self, sample):

        img, pose, scr = sample['image'], sample['pose'], sample['scr']
        return {'image': (img-self.mean)/self.divider, 
                'scr': scr,
                'pose': pose}


class fix_crop(object):
    def __init__(self, validROI_x=16, validROI_y=8, validROI_w=592, validROI_h=336):
        self.validROI_x = validROI_x
        self.validROI_y = validROI_y
        self.validROI_w = validROI_w
        self.validROI_h = validROI_h
    def __call__(self, sample):
        crop_size_w, crop_size_h = self.validROI_w, self.validROI_h
        xoffs, yoffs = self.validROI_x, self.validROI_y
        if 'scr' in sample:
            img, pose, scr = sample['image'], sample['pose'], sample['scr']
            return {'image': img[:,yoffs:yoffs+crop_size_h,xoffs:xoffs+crop_size_w],
                    'scr': scr[:,yoffs:yoffs+crop_size_h,xoffs:xoffs+crop_size_w],
                'pose': pose}
        else:
            img, pose = sample['image'], sample['pose']
            return {'image': img[:,yoffs:yoffs+crop_size_h,xoffs:xoffs+crop_size_w],
                'pose': pose}


def maskedPnorm(Y, Y_true, ptp=ptp, p=2, bg=mean_sc):
    err = torch.norm(torch.FloatTensor(ptp).to(device) * (Y.permute(0,2,3,1)-Y_true.permute(0,2,3,1)), p=p, dim=3)
    mask = torch.norm(Y_true.permute(0,2,3,1) - torch.FloatTensor(bg).to(device), p=p, dim=3) >= 0.00001
    err = err[mask]
        
    return torch.mean(err)

def robust_maskedPnorm(Y, Y_true, ptp=ptp, p=2, bg=mean_sc, mask_flag=True):
    ptp = torch.FloatTensor(ptp).to(device)
    err = torch.abs(ptp * (Y.permute(0,2,3,1)-Y_true.permute(0,2,3,1)))
    
    c1, c2, c3 = ptp / 2.0
    
    r1 = err[:,:,:,0]
    maks1 = r1 <= c1
    r1[maks1] = c1**2 / 6 * (1 - (1 - (r1[maks1] / c1)**2)**3)
    r1[~maks1] = c1**2 / 6
    
    r2 = err[:,:,:,1]
    maks2 = r2 <= c2
    r2[maks2] = c2**2 / 6 * (1 - (1 - (r2[maks2] / c2)**2)**3)
    r2[~maks2] = c2**2 / 6
    
    r3 = err[:,:,:,2]
    maks3 = r3 <= c3
    r3[maks3] = c3**2 / 6 * (1 - (1 - (r3[maks3] / c3)**2)**3)
    r3[~maks3] = c3**2 / 6
    
    err = r1 + r2 + r3
    
    if mask_flag:
        mask = torch.norm(Y_true.permute(0,2,3,1) - torch.FloatTensor(bg).to(device), p=p, dim=3) >= 0.00001
        err = err[mask]
        
    return torch.mean(err)


class RunningMedian:
    def __init__(self):
        self.left, self.right = [float('inf')], [float('inf')]
    def peek_left(self):
        return -self.left[0]
    def pop_right(self):
        return heapq.heappop(self.right)
    def pop_left(self):
        return -heapq.heappop(self.left)
    def peek_right(self):
        return self.right[0]
    def pushpop_left(self, n):
        return -heapq.heappushpop(self.left, -n)
    def pushpop_right(self, n):
        return heapq.heappushpop(self.right, n)
    def push_left(self, n):
        heapq.heappush(self.left, -n)
    def push_right(self, n):
        heapq.heappush(self.right, n)
    def size(self):
        return len(self.left)+len(self.right)-2
    def update(self, n):
        median = self.median()
        if median is None:
             self.push_left(n)
        elif n > median:
            self.push_right(n)
        else:
            self.push_left(n)
        while len(self.left)-len(self.right) > 1:
            self.push_right(self.pop_left())
        while len(self.right)-len(self.left) > 1:
            self.push_left(self.pop_right())  
    def median(self):
        if len(self.left)==1 and len(self.right)==1: 
            return None
        if len(self.left) == 1:
            # print('left_empty')
            return self.peek_right()
        if len(self.right) == 1:
            # print('right_empty')
            return self.peek_left()
        if (len(self.left)%2 + len(self.right)%2) % 2 == 0:
            # print('same parity')
            return (self.peek_left() + self.peek_right())/2
        # print('opposite parity', end = ',\t')
        if len(self.left) > len(self.right):
            # print('left longer')
            return self.peek_left()
        # print('right longer')
        return self.peek_right()

def get_median(scr_hats, Ts):
    T_hats = []
    t_error_list = []
    r_error_list = []
    t_median = RunningMedian()
    r_median = RunningMedian()
    pbar = tqdm(zip(scr_hats, Ts), total=len(Ts))

    with mp.Pool() as pool:
        for res in pool.imap_unordered(map_fn, enumerate(pbar)):
            i, T_hat, ang_err, lin_err, inliers = res
            t_error_list.append(lin_err)
            r_error_list.append(ang_err)
            r_median.update(ang_err)
            t_median.update(lin_err)
            T_hats.append((i, T_hat))
            assert t_median.size() == len(T_hats), f'median.size() ({median.size()}) != len(T_hats) ({len(T_hats)})'
    return r_median.median(), t_median.median()


def validate(net, dataloader):
    net.eval()
    scr_predicts, true_poses = [], []
    scr_labels = []

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        inputs, scrs, poses = sample_batched['image'], sample_batched['scr'], sample_batched['pose']
        inputs = inputs.to(device, dtype=torch.float)
        with torch.no_grad():
            scr_predict = net(inputs)[1]

        scr_predicts.append(scr_predict.cpu().numpy())
        scr_labels.append(torch.nn.functional.interpolate(scrs.cpu(), scale_factor=1/8.0, mode='bilinear').numpy())
        true_poses.append(poses.numpy())
        
    scr_predicts = np.concatenate(scr_predicts)
    scr_labels = np.concatenate(scr_labels)
    true_poses = np.concatenate(true_poses)
    return scr_predicts, scr_labels, true_poses

def map_fn(args):
    scale = 8
    i, (scr, true_pose) = args
    scr = scr.transpose(1,2,0)
    predict_pose, mask, inliers = compute_pose(scr, xofs=16/scale, yofs=8/scale, K=K/scale, ptp_min=ptp_min, 
                                               ptp=ptp, background_color=mean_sc, PnPSolver=cv2.solvePnPRansac,
                                               bg_eps=0.07, pnp_method=cv2.SOLVEPNP_EPNP,
                                               pnp_kwargs = {'iterationsCount': 1000, 'reprojectionError': 3.1})
    ang_err = compute_angle(predict_pose[:3,:3], true_pose[:3,:3])
    lin_err = compute_translation(predict_pose[:3,3], true_pose[:3,3])
    return i, predict_pose, ang_err, lin_err, inliers

def compute_mean_error(labels, predicts):
    labels = labels.transpose(0,2,3,1)
    predicts = np.clip(predicts.transpose(0,2,3,1), a_min=0.0,a_max=1.0)
    
    return np.mean(np.linalg.norm(labels-predicts, ord=2, axis=-1))

