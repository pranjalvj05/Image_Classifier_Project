#!/usr/bin/env python3
#predict_args.py


import argparse

#argparse parser
#https://pymotw.com/3/argparse

def get_args():
   
    parser = argparse.ArgumentParser(
        description="Image prediction.",
        usage="python ./predict.py /path/to/image.jpg checkpoint.pth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('path_to_image',
                        help='Path to image file.',
                        action="store")

    parser.add_argument('checkpoint_file',
                        help='Path to checkpoint file.',
                        action="store")

    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='Directory to save training checkpoint file',
                        )

    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int,
                        help='Return top KK most likely classes.',
                        )

    parser.add_argument('--category_names',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to file containing the categories.',
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Use GPU')

    parser.parse_args()
    return parser

#main
def main():
    
    print(f'Command line argument utility for predict.py.\nTry "python train.py -h".')


if __name__ == '__main__':
    main()
