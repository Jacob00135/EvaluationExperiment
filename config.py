import os


def cntpath(p1, p2):
    p = os.path.realpath(os.path.join(p1, p2))
    if os.path.isdir(p) and p[-1] != '/':
        p = p + '/'
    return p


# region 路径相关
root_path = os.path.realpath(os.path.dirname(__file__))
checkpoint_dir_path = cntpath(root_path, 'checkpoint_dir')
tb_log_path = cntpath(root_path, 'tb_log')
late_embeddings_path = cntpath(root_path, 't_SNE/late_embeddings')
mid_embeddings_path = cntpath(root_path, 't_SNE/mid_embeddings')
early_embeddings_path = cntpath(root_path, 't_SNE/early_embeddings')
data_2_sq_path = cntpath(root_path, 'data_2/sq')
mri_csv_path = cntpath(root_path, 'lookupcsv/CrossValid/no_cross')

mri_path = '/home/lxs/ADNI/npy/'
# endregion

# region 常量
category_map = {
    'NC': 0,
    'MCI': 1,
    'DE': 2
}
category_list = ['NC', 'MCI', 'DE']
feature_columns = [
    'faq_EVENTS',
    'faq_GAMES',
    'faq_SHOPPING',
    'faq_TRAVEL',
    'faq_MEALPREP',
    'trailB',
    'trailA',
    'faq_BILLS',
    'faq_REMDATES',
    'faq_TAXES',
    'digitBL',
    'digitB',
    'digitF',
    'digitFL',
    'lm_del',
    'lm_imm',
    'gds',
    'mmse',
    'boston',
    'animal',
    'age'
]
required_columns = ['RID', 'VISCODE', 'COG'] + feature_columns
# endregion


def warning_print(string):
    print("\033[0;31;40m", string, "\033[0m")
