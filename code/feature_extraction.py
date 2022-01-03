import torch
import os
import numpy as np
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale
from PIL import Image
import scipy

def features(args, image, features_path, count):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #Creating CNN model
    model = D2Net(
        model_file= args.model_file,
        use_relu=args.use_relu,
        use_cuda=use_cuda
    )

    #Process the file
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)
    
    resized_image = image
    [w, h] = image.shape[: 2]
    size = args.max_edge/max(resized_image.shape)
    if max(resized_image.shape) > args.max_edge:
        # size = tuple(np.array(h * args.max_edge/max(resized_image.shape), w * args.max_edge/max(resized_image.shape)).astype(int))
        resized_image = np.array(Image.fromarray(resized_image).resize((int(args.max_edge), int(args.max_edge)), Image.BILINEAR))
    
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        size = tuple(np.array(args.max_sum_edges/sum(resized_image.shape[: 2])).astype(int))
        resized_image = np.array(Image.fromarray(resized_image).resize(size, Image.BILINEAR))
    
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(torch.tensor(image[np.newaxis, :, :, :].astype(np.float32),device=device),model)
        else:
            keypoints, scores, descriptors = process_multiscale(torch.tensor(image[np.newaxis, :, :, :].astype(np.float32), device=device), model, scales=[1])

    #Input Image Coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    keypoints = keypoints[:, [1, 0, 2]]
    if args.output_type == 'npz':
        with open(os.path.join(features_path,'features')+str(count) + args.output_extension, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints = keypoints,
                scores = scores,
                descriptors = descriptors
            )
    elif args.output_type == 'mat':
        with open(os.path.join(features_path, 'features')+str(count) + args.output_extension, 'wb') as output_file:
            scipy.io.savemat(
                output_file,
                {
                    'keypoints': keypoints,
                    'scores': scores,
                    'descriptors': descriptors
                }
            )
    else:
        raise ValueError('Unknown output type.')