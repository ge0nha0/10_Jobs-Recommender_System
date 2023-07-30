from dataloader import dataloader_factory


from options import args

def main():

    train, val, test = dataloader_factory(args)
    print('dataloader complete')

if __name__ == '__main__':
    main()