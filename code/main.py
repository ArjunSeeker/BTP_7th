import os
import argparse

from skimage.feature import match
from feature_extraction import *
from match import *

def main(args):
    # path for input image 
    input_image_path = 'image_input'
    if not os.path.exists(input_image_path):
        os.mkdir(input_image_path)
    input_image_path = os.path.join('image_input')
    
    # path for image feature extraction
    image_features_path = 'features_extracted'
    if not os.path.exists(image_features_path):
        os.mkdir(image_features_path)
    image_features_path = os.path.join('features_extracted')
    
    # appending the iput images into the list
    input_images_list = []
    for image_path in os.listdir(input_image_path):
        path = os.path.join(input_image_path, image_path)
        input_images_list.append(cv2.imread(path))
    len_input_image_list = len(input_images_list)
    count = 1
    
    image1 = input_images_list[0]
    features(args, image1, image_features_path, count)
    feat1 = np.load(os.path.join(image_features_path,'features')+str(count)+args.output_extension)
    count +=1
    
    image2 = input_images_list[1]
    features(args, image2, image_features_path, count)
    feat2 = np.load(os.path.join(image_features_path,'features')+str(count)+args.output_extension)
    stitch_images(args, image1, feat1, image2, feat2, count, len_input_image_list)
    count += 1
    SI_path = os.path.join('stitched_images','SI')
    for index in range(2,len(input_images_list)):
        print(count)
        image1 = cv2.imread(SI_path+str(count)+'.jpg')
        features(args, image1, image_features_path, count)
        feat1 = np.load(os.path.join(image_features_path,'features') + str(count) + args.output_extension)
        count += 1
        
        image2 = input_images_list[index]
        features(args, image2, image_features_path, count)
        feat2 = np.load(os.path.join(image_features_path,'features') + str(count) + args.output_extension)
        stitch_images(args, image1, feat1, image2, feat2, count, len_input_image_list)
        count += 1






def parseArg():
    parser = argparse.ArgumentParser(description='Feature extraction script')
    parser.add_argument(
        '--image_list_file', type=str, required=False,
        help='path to a file containing a list of images to process'
    )

    parser.add_argument(
        '--preprocessing', type=str, default='torch',
        help='image preprocessing (caffe or torch)'
    )
    parser.add_argument(
        '--model_file', type=str, default='code/models/d2_tf_no_phototourism.pth',
        help='path to the full model'
    )

    parser.add_argument(
        '--max_edge', type=int, default=1600,
        help='maximum image size at network input'
    )
    parser.add_argument(
        '--max_sum_edges', type=int, default=2800,
        help='maximum sum of image sizes at network input'
    )

    parser.add_argument(
        '--output_extension', type=str, default='.npy',
        help='extension for the output'
    )
    parser.add_argument(
        '--output_type', type=str, default='npz',
        help='output file type (npz or mat)'
    )
    parser.add_argument(
        '--image_output_extension', type = str, default='.jpg',
        help = 'extension for the output image'
    )
    parser.add_argument(
        '--multiscale', dest='multiscale', action='store_true',
        help='extract multiscale features'
    )
    parser.set_defaults(multiscale=False)

    parser.add_argument(
        '--no-relu', dest='use_relu', action='store_false',
        help='remove ReLU after the dense feature extraction module'
    )
    parser.set_defaults(use_relu=True)
    args = parser.parse_args()
    print(args)
    return args
if __name__ == "__main__":
    args = parseArg()
    main(args)