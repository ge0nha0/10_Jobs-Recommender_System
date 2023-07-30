def set_template(args):
    if args.mode is None:
        return
    
    elif args.mode.startswith('train'):
        args.device = 'cuda' # 'cpu'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'AdamW'
        args.lr = 0.00005
        args.weight_decay = 0
        args.momentum = None
        args.num_epochs = 5

        args.dataloader_random_seed = 0
        batch = 64
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.model_init_seed = 0

        args.max_len = 128
        args.hidden_units = 768
        args.dropout = 0.1


        args.model_version = 'original'
        args.sent_idx = 0
        args.label_idx = 1
        
    
    elif args.mode.startswith('test'):
        return
    
    elif args.mode.startswith('recommend'):
        return