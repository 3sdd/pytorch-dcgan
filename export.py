import argparse
import os
import torch
from models import Generator


def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generator-path',type=str,default='./results/g.pth')
    parser.add_argument('--out-path',type=str,default='./results/dcgan-generator.onnx')
    parser.add_argument('--verbose',action='store_true' )
    return parser

def export():
    args=get_parser().parse_args()

    os.makedirs(os.path.dirname(args.out_path),exist_ok=True)

    dummy_input = torch.randn(1,100,1,1)
    model = Generator()
    model.load_state_dict(torch.load(args.generator_path,map_location='cpu'))

    input_names = [ "input" ]
    output_names = [ "output" ]

    torch.onnx.export(model, dummy_input, args.out_path, verbose=args.verbose, input_names=input_names, output_names=output_names)
    print("finish")

if __name__=="__main__":
    export()