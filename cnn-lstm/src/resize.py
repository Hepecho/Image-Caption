import argparse
import os
from PIL import Image
 

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    os.makedirs(output_dir, exist_ok=True)
    images = os.listdir(image_dir)
    for image in images:
        image_path = os.path.join(image_dir, image)
        with Image.open(image_path) as img:
            img_ = img.resize(size)
        img_.save(os.path.join(output_dir, image))
    

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/train2014/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='data/resized2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)