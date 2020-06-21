import numpy as np
import cv2
import time
import matplotlib.image
import argparse
import os

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.radius = 3 * self.sigma_s
    def joint_bilateral_filter(self, image, guidance):
        height = image.shape[0]
        width = image.shape[1]
        channel = image.shape[2]
        filtered = np.zeros(image.shape)
        for x in range(width):
            for y in range(height):

                # edges processing
                y_bottom    = np.maximum(0,      y - self.radius)
                y_top       = np.minimum(height, y + self.radius + 1)
                x_left      = np.maximum(0,      x - self.radius)
                x_right     = np.minimum(width,  x + self.radius + 1)
 
                # h_space: size = (2r + 1) x (2r + 1) (window size)
                p_q = [[i**2 + j**2 for i in range(x_left-x, x_right-x)] for j in range(y_bottom-y, y_top-y)]
                Gs = gaussian(p_q, self.sigma_s)
                Jp_Jq = np.zeros((y_top - y_bottom ,x_right - x_left))
                for c in range(channel):
                    Jp = image[y][x][c]
                    if guidance is not None:
                        # input guide image is gray image
                        Jp_Jq = (Jp - guidance[y_bottom:y_top, x_left:x_right]) ** 2
                    else:
                        # input guide image is itself
                        Jp_Jq = (Jp - image[y_bottom:y_top, x_left:x_right, c]) ** 2

                    Gr = gaussian(Jp_Jq, self.sigma_r )
                    Iq = image[y_bottom:y_top, x_left:x_right,c]
                    Gs_Gr = np.multiply(Gs, Gr)
                    # sum( Gs * Gr * Iq )/sum( GS * Gr )
                    filtered[y, x, c] = np.sum(np.multiply(Gs_Gr, Iq))/ np.sum(Gs_Gr)
                    
        filtered *= 255
        return filtered
    
def create_folder(path):
    if path[-1]!='/':
        path = path + '/'
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory " , path ,  " Created ")
    else:
        print("Directory " , path ,  " already exists")
    return path

def gaussian(x,sigma):
    return np.exp(-np.array(x) / (2 * (sigma ** 2))) 

def generateGrayImage(img_rgb,Wr,Wg,Wb):
    return ((img_rgb[:,:,0] * Wr + img_rgb[:,:,1] * Wg + img_rgb[:,:,2] * Wb) / 255.0)

def generateCandidateWeight():
    weight = []
    for r in range(11):
        for g in range(11):
            for b in range(11):
                if((r+g+b)==10):
                    weight.append((r / 10.0, g / 10.0 ,b / 10.0 ))
    return weight

def Euclidean_distance(main_coord, other_coord):
    distance = 0
    for i in range(np.array(main_coord).shape[0]):
        distance += (main_coord[i] - other_coord[i])**2
    return np.sqrt(distance)



def main(args):
    output_path = create_folder(args.output)
    print('mode:' + args.mode)
    input_path = args.inputfolder
    if args.mode=='a':
        for dirname, _, filenames in os.walk(input_path):
            for img_name in filenames:
                # load img
                img = cv2.imread(os.path.join(input_path, img_name))
                print(os.path.join(input_path, img_name))
                img_rgb_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # set parameter
                sigma_s_g = [1,2,3]
                sigma_r_g = [0.05,0.1,0.2]
                w = generateCandidateWeight()
                resize_factor = 0.1
                # speed up, resize img
                img_rgb = cv2.resize(img_rgb_ori, None, fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_CUBIC)
                guidance = cv2.resize(guidance, None, fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_CUBIC)
                
                final_vote = dict()
                for wi in w:
                    final_vote[wi] = 0
                    
                # transfer to [0,1]
                image = img_rgb / 255.0
            
                for sigma_s in sigma_s_g:
                    for sigma_r in sigma_r_g: 
                        print('processing sigma_s:%d sigma_r:%.2f' % (sigma_s, sigma_r))
                        # determine JBF class
                        JBF = Joint_bilateral_filter(sigma_s, sigma_r)
                        # generate BFimage 
                        bf_out = JBF.joint_bilateral_filter(image, guidance = None)
                        #save cost result
                        Wi_dict = dict() 
                        for wi in range(len(w)):
                            img_gray = generateGrayImage(img_rgb, w[wi][0], w[wi][1], w[wi][2])
                            jbf_out_byGray = JBF.joint_bilateral_filter(image, guidance = img_gray)
                            #cost function
                            jbf_error = np.sum(np.abs(bf_out - jbf_out_byGray))
                            Wi_dict[w[wi]] = jbf_error
                        # save vote 
                        Wi_vote = dict()
                        for wi in w:
                            w_error = Wi_dict[wi]
                            dist_T = 0.3
                            wi_neighbor = []
                            # find neighbor
                            for Wg_i in range(len(w)):
                                # using Euclidean distance
                                dist = Euclidean_distance(wi, w[Wg_i])
                                if dist <=dist_T and not dist==0:
                                    wi_neighbor.append(w[Wg_i])
                            # vote local minima
                            vote = 0
                            total_neighbor = len(wi_neighbor)
                            for neighbor_i in range(total_neighbor):
                                neighbor_error = Wi_dict[wi_neighbor[neighbor_i]]
                                if neighbor_error > w_error:
                                    vote +=1
                            Wi_vote[wi] = (vote,total_neighbor)
                        #  find local minima
                        for wi in w:
                            vote_result = Wi_vote[wi]
                            vote_rate = vote_result[0] / vote_result[1]
                            if(vote_rate == 1):
                                final_vote[wi]+=1
                                
                # find top n vote weight and save img
                n = 3
                c = 0
                for k, v in sorted(final_vote.items(), key=lambda item: item[1],reverse = True):
                    if c<n:
                        img_gray = generateGrayImage(img_rgb_ori,k[0],k[1],k[2])
                        save_file_name = 'Wr_' + str(k[0]) +' Wg_'+str(k[1]) + ' Wb_'+ str(k[2]) + '_vote_' + str(v) + '_' + img_name
                        matplotlib.image.imsave(output_path + save_file_name, img_gray, cmap='gray')
                    else:
                        break
                    c+=1
    elif args.mode =='c':
        for dirname, _, filenames in os.walk(input_path):
            for img_name in filenames:
                print(os.path.join(input_path, img_name))
                img_rgb = cv2.cvtColor(cv2.imread(os.path.join(input_path, img_name)), cv2.COLOR_BGR2RGB)
                img_gray = generateGrayImage(img_rgb, 0.299, 0.587 , 0.114)
                matplotlib.image.imsave(output_path + 'gray_' + img_name, img_gray, cmap='gray')
    else:
        raise NotImplementedError("Please specify the mode \"a\" for advanced or \"c\" for conventional method.")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfolder", default="testdata/",
                        help="input folder processing all image")
    parser.add_argument("-o", "--output", default="output",
                        help="output directory")
    parser.add_argument("--mode", default="a",
                        help="c: conventional; a: advanced")
    start_time = time.time()
    main(parser.parse_args())
    
    print("--- %s seconds ---" % (time.time() - start_time))