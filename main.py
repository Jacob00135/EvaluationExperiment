import pdb
import json
from model_wrappers import Multask_Wrapper
from nonImg_model_wrappers import NonImg_Model_Wrapper, Fusion_Model_Wrapper
from utils import read_json, plot_shap_bar, plot_shap_heatmap, plot_shap_beeswarm
from performance_eval import whole_eval_package
from multiprocessing import Process
import os
from config import root_path, checkpoint_dir_path, tb_log_path, late_embeddings_path, mid_embeddings_path, \
    early_embeddings_path, data_2_sq_path, mri_path
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


def multi_process(main_config, task_config, tid_gpu, tasks):
    """
    :param tid_gpu: python dictionary {task id : GPU id} for example run cross0 on gpu 1, run cross1 on gpu2 {0:1, 1:2}
    """
    model_name = main_config['model_name']
    csv_dir = main_config['csv_dir']
    processes = []
    for i in tid_gpu:
        main_config['csv_dir'] = csv_dir + 'cross{}/'.format(i)
        main_config['model_name'] = model_name + '_cross{}'.format(i)
        p = Process(target=main, args=(main_config, task_config, tid_gpu[i], tasks))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def crossValid(main_config, task_config, device, tasks):
    model_name = main_config['model_name']
    main_config['csv_dir'] = os.path.realpath(os.path.join(root_path, main_config['csv_dir']))
    model = Multask_Wrapper(
        tasks=tasks,
        device=device,
        main_config=main_config,
        task_config=task_config,
        seed=1000
    )
    model.train()
    thres = model.get_optimal_thres()
    model.gen_embd(['test'], layer='mid')
    model.gen_score(['train', 'valid', 'test'], thres)

    # region 五折交叉检验
    """
    for i in range(5):
        main_config['csv_dir'] = os.path.realpath(os.path.join(csv_dir, 'cross{}/'.format(i)))
        main_config['model_name'] = model_name + '_cross{}'.format(i)
        model = Multask_Wrapper(tasks=tasks,
                                device=device,
                                main_config=main_config,
                                task_config=task_config,
                                seed=1000)
        model.train()
        thres = model.get_optimal_thres()
        model.gen_embd(['test'], layer='mid')
        #model.gen_embd(['exter_test'], layer='mid')
        model.gen_score(['train', 'valid', 'test'], thres)
        #model.shap_mid()
        #model.shap_mid(task_idx=1, path='/data_2/NACC_ALL/ADNI_neuroPath/', file='ADNI_NP.csv')
    """
    # endregion

    whole_eval_package(
        model_name,
        'test',
        os.path.abspath(os.path.join(tb_log_path, model_name, 'test_perform'))
    )
    whole_eval_package(
        model_name,
        'valid',
        os.path.abspath(os.path.join(tb_log_path, model_name, 'valid_perform'))
    )
    print('运行完毕：{}'.format(os.path.basename(__file__)))


def fusion_CNN_main(main_config, task_config, tasks):
    model_name = main_config['model_name']
    csv_dir = os.path.realpath(os.path.join(root_path, main_config['csv_dir']))
    for i in range(5):
        processes = []
        for j in range(4):
            main_config['csv_dir'] = csv_dir + 'cross{}/'.format(i) + 'fold{}/'.format(j)
            main_config['model_name'] = model_name + '_cross{}_fold{}'.format(i, j)
            p = Process(target=crossValid_Fusion, args=(main_config, task_config, j, tasks))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


def crossValid_nonImg(main_config, task_config, tasks, shap_analysis=False):
    model_name = main_config['model_name']
    csv_dir = main_config['csv_dir']
    shap, data = [], []
    for i in range(5):
        main_config['csv_dir'] = csv_dir + 'cross{}/'.format(i)
        main_config['model_name'] = model_name + '_cross{}'.format(i)
        model = NonImg_Model_Wrapper(tasks=tasks,
                                     main_config=main_config,
                                     task_config=task_config,
                                     seed=1000)
        model.train()
        thres = model.get_optimal_thres()
        model.gen_score(['train', 'valid', 'test'], thres)
        if shap_analysis:
            shap_values, features_values = model.shap('test_shap')
            shap.append(shap_values)
            data.append(features_values)
    whole_eval_package(
        model_name,
        'test',
        os.path.abspath(os.path.join(tb_log_path, '{}_cross0/test_perform'.format(model_name)))
    )
    # whole_eval_package(model_name, 'test', '/home/huangyunyou/ncomms2022-main/tb_log/{}_cross0/test_perform'.format(model_name))
    # whole_eval_package(model_name, 'OASIS')
    if shap_analysis:
        plot_shap_bar(
            os.path.abspath(os.path.join(tb_log_path, '{}_cross0/'.format(model_name))),
            model_name,
            'test_shap',
            tasks,
            top=15
        )
        # plot_shap_bar('/home/huangyunyou/ncomms2022-main/tb_log/' + model_name + '_cross0/', model_name, 'test_shap', tasks, top=15)
        plot_shap_beeswarm(
            os.path.abspath(os.path.join(tb_log_path, '{}_cross0/'.format(model_name))),
            shap,
            data,
            tasks,
            'test_shap'
        )
        # plot_shap_beeswarm('/home/huangyunyou/ncomms2022-main/tb_log/' + model_name + '_cross0/', shap, data, tasks, 'test_shap')


def crossValid_Fusion(main_config, task_config, tasks, shap_analysis=True):
    model_name = main_config['model_name']
    csv_dir = main_config['csv_dir']
    shap, data = [], []
    for i in range(5):
        main_config['csv_dir'] = csv_dir + 'cross{}/'.format(i)
        main_config['model_name'] = model_name + '_cross{}'.format(i)
        model = Fusion_Model_Wrapper(tasks=tasks,
                                     main_config=main_config,
                                     task_config=task_config,
                                     seed=1000)
        model.train()
        thres = model.get_optimal_thres(csv_name='valid')
        model.gen_score(['test'], thres)
        #model.gen_score(['neuro_test_mri'], thres)
        #if shap_analysis:
            #shap_values, features_values = model.shap('test_shap')
            #shap.append(shap_values)
            #data.append(features_values)
    whole_eval_package(model_name, 'test', 'tb_log/{}_cross0/test_perform'.format(model_name))
    # whole_eval_package(model_name, 'OASIS_mri', 'tb_log/{}_cross0/OASIS_perform'.format(model_name))
    #whole_eval_package(model_name, 'neuro_test_mri', 'tb_log/{}_cross0/neuro_test_perform'.format(model_name))
    #if shap_analysis:
        #plot_shap_bar('/home/huangyunyou/ncomms2022-main/tb_log/'+model_name+'_cross0/', model_name, 'test_shap', tasks, top=15)
        #plot_shap_beeswarm('/home/huangyunyou/ncomms2022-main/tb_log/'+model_name+'_cross0/', shap, data, tasks, 'test_shap')


def printf_dict(d):
    print(json.dumps(d, indent=4, ensure_ascii=True))


if __name__ == "__main__":

    # main(tasks = ['COG', 'ADD'],
    #      device = 2,
    #      main_config = read_json('config.json'),
    #      task_config = read_json('task_config.json'))

    crossValid(tasks = ['COG'],
            device = 0,
            main_config = read_json('config.json'),
            task_config = read_json('task_config.json'))

    # multi_process(tasks = ['COG', 'ADD'],
    #               tid_gpu = {0:0, 1:1, 2:2},
    #               main_config = read_json('config.json'),
    #               task_config = read_json('task_config.json'))

    #fusion_CNN_main(main_config = read_json('config.json'),
    #        task_config = read_json('task_config.json'),
    #        tasks = ['COG', 'ADD'])
    #crossValid_nonImg(tasks=['COG'],
            #main_config=read_json('config.json'),
            #task_config=read_json('nonImg_task_config.json'))
    
    # crossValid_Fusion(main_config = read_json('config.json'),
    #         task_config = read_json('nonImg_task_config.json'),
    #         tasks = ['COG'])


    # plot_shap_heatmap(['CatBoost', 'XGBoost', 'RandomForest', 'DecisionTree', 'SupportVector', 'NearestNeighbor', 'Perceptron'],
    #                   ['COG', 'ADD'], 'test_shap')





"""
ssh -L 16005:127.0.0.1:6006 username@155.41.207.229
"""

