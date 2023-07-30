from templates import set_template
import argparse


parser = argparse.ArgumentParser(description='RecPlay')

## main ##
parser.add_argument('--mode', type=str, default='train', choices=['train','test','recommend'])

## collection ##
parser.add_argument('--end_point', type=str, default='http://openapi.work.go.kr/opi/opi/opia/wantedApi.do?')
parser.add_argument('--service_key', type=str, default='WNLHPWTG5WE1T0XSSSWCL2VR1HK')
parser.add_argument('--call_tp', type=str, default='L',choices=['L','D'])
parser.add_argument('--return_type', type=str, default='XML')

## Dataset ##
parser.add_argument('--sent_idx', type=int, default=0)
parser.add_argument('--label_idx', type=int, default=1)
parser.add_argument('--pad', type=bool, default=True)
parser.add_argument('--pair', type=bool, default=False)

## DataLoader ##
parser.add_argument('--dataloader_random_seed', type=int, default=0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)


## Train ##
# Device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate') 
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# epochs #
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')


## Model ##
################
parser.add_argument('--model_init_seed', type=int, default=0) # default=None

## import model ##
parser.add_argument('--model_version', type=str, default='original',choices=['original','fine_tuned']) # 버전별로 선택할 수 있게 추가 작업

# BERT # 
parser.add_argument('--max_len', type=int, default=128, help='Length of sequence for bert')
parser.add_argument('--hidden_units', type=int, default=768, help='Size of hidden vectors (d_model)')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')

args = parser.parse_args(args=[])
# args = parser.parse_args()
set_template(args)
