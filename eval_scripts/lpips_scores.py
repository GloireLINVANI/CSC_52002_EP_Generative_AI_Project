import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data_dir', type=str, default='./samples', help='directory containing datasets')
parser.add_argument('-o', '--output', type=str, default='lpips_scores.csv', help='output CSV file')
parser.add_argument('-v', '--version', type=str, default='0.1', help='LPIPS version')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

# Initializing the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if opt.use_gpu:
    loss_fn.cuda()

# Write the header to the CSV file
with open(opt.output, 'w') as f:
    f.write("dataset,comparison,LPIPS\n")

# Loop through each dataset and mask
for dataset in os.listdir(opt.data_dir):
    dataset_path = os.path.join(opt.data_dir, dataset)
    if os.path.isdir(dataset_path):
        # Define the subfolders
        subfolders = ["base", "repaint", "rpip"]

        # Run the LPIPS calculations and append the results to the CSV file
        for subfolder in subfolders:
            original_dir = os.path.join(dataset_path, "original")
            compare_dir = os.path.join(dataset_path, subfolder)
            if os.path.isdir(original_dir) and os.path.isdir(compare_dir):
                files = os.listdir(original_dir)
                for file in files:
                    if os.path.exists(os.path.join(compare_dir, file)):
                        # Load images
                        img0 = lpips.im2tensor(lpips.load_image(os.path.join(original_dir, file)))  # RGB image from [-1,1]
                        img1 = lpips.im2tensor(lpips.load_image(os.path.join(compare_dir, file)))

                        if opt.use_gpu:
                            img0 = img0.cuda()
                            img1 = img1.cuda()

                        # Compute distance
                        dist01 = loss_fn.forward(img0, img1)
                        with open(opt.output, 'a') as f:
                            f.write(f"{dataset},{subfolder},{dist01.item():.6f}\n")
