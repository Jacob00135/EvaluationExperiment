import os
import pdb
from utils import read_json
from config import root_path
from nonImg_model_wrappers import NonImg_Model_Wrapper


def main():
    model = NonImg_Model_Wrapper(
        tasks=['COG'],
        main_config={
            'model_name': 'nonimg_model_20231124',
            'csv_dir': os.path.realpath(os.path.join(root_path, 'data_preprocess/dataset'))
        },
        task_config=read_json('nonImg_task_config.json'),
        seed=1000
    )
    model.train()


if __name__ == '__main__':
    main()
