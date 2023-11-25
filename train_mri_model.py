import os
import pdb
from config import root_path
from utils import read_json
from model_wrappers import Multask_Wrapper


def main():
    model = Multask_Wrapper(
        tasks=['COG'],
        device=0,
        main_config={
            'model_name': 'MRI_model_20231124',
            'csv_dir': os.path.realpath(os.path.join(root_path, 'data_preprocess/dataset'))
        },
        task_config=read_json('task_config.json'),
        seed=1000
    )
    model.train()


if __name__ == '__main__':
    main()
