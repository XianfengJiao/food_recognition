import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='model parameters')
    # general
    parser.add_argument('--seed', default=2019, type=int)
    parser.add_argument('--workers', default=8, type=int)

    # data
    parser.add_argument('--final_img_path', default='./finalImages_finetune/')  # missing
    parser.add_argument('--final_img_ori_path', default='./mini_food_data/FoodImages/')
    parser.add_argument('--final_img_verb_path', default='./finalImages_finetune_verb/')  # missing
    parser.add_argument('--step_img_path', default='./mini_food_data/StepImages/')  # missing data
    parser.add_argument('--recipe_path', default='./mini_food_data/recipe_feat_new/')

    parser.add_argument('--train_lis', default='./mini_food_data/data_90/train_lis.pkl')
    parser.add_argument('--valid_lis', default='./mini_food_data/data_90/valid_lis.pkl')
    parser.add_argument('--test_lis', default='./mini_food_data/data_90/test_lis.pkl')
    parser.add_argument('--test_lis_split', default='./mini_food_data/data_90/test_lis_split.pkl')
    parser.add_argument('--classes', default='./mini_food_data/data_90/class.pkl')

    parser.add_argument('--VireoFood172', default='./mini_food_data/data_90/VireoFood172_finetune_path_list.pkl')
    parser.add_argument('--recipe_dict', default='./mini_food_data/data_90/recipeDict.pkl')
    parser.add_argument('--name_dict', default='./mini_food_data/data_90/nameDict.pkl')
    parser.add_argument('--word_mat', default='./mini_food_data/data_90/word_mat.npy')

    # im2recipe model
    parser.add_argument('--embDim', default=1024, type=int)
    parser.add_argument('--numLayer', default=3, type=int)
    parser.add_argument('--numHeads', default=8, type=int)

    parser.add_argument('--titleDim', default=64, type=int)
    parser.add_argument('--ingrDim', default=64, type=int)
    parser.add_argument('--wordDim', default=64, type=int)
    parser.add_argument('--wordModelDim', default=256, type=int)
    parser.add_argument('--imageDim', default=2048, type=int)

    # about the batch
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--semantic_pc', default=0.4, type=float)
    parser.add_argument('--same_class_num', default=2, type=int)

    parser.add_argument('--semantic_reg', default=True, type=bool)
    parser.add_argument('--numClasses', default=1026, type=int)
    parser.add_argument('--sem_weight', default=0.1, type=float)
    parser.add_argument('--cos_weight', default=1.0, type=float)

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=720, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--valfreq', default=1, type=int)

    # dataset
    parser.add_argument('--titleMaxlen', default=20, type=int)
    parser.add_argument('--ingrMaxlen', default=30, type=int)
    parser.add_argument('--wordMaxlen', default=400, type=int)  # total word
    parser.add_argument('--stepMaxlen', default=25, type=int)
    parser.add_argument('--wpsMaxlen', default=100, type=int)  # word per step
    parser.add_argument('--imageMaxlen', default=60, type=int)

    # MedR / Recall@1 / Recall@5 / Recall@10
    parser.add_argument('--embtype', default='image', type=str)  # [image|recipe] query type
    parser.add_argument('--medr', default=1000, type=int)

    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--startTest', default=50, type=int)
    parser.add_argument('--freeImage', default=False, type=bool)
    parser.add_argument('--freeRecipe', default=True, type=bool)
    parser.add_argument('--checkpoint', default='./ckpt/', type=str)
    parser.add_argument('--maxCkpt', default=3, type=int)
    parser.add_argument('--restore', default='', type=str)
    # parser.add_argument('--restore', default='./ckpt-trans/model_e052_v-2.000.pth.tar', type=str)

    # test
    parser.add_argument('--path_results', default='results/', type=str)

    return parser
