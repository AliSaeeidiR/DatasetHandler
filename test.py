import os
from datasethandler.DatasetTypes import Classification
from datasethandler.Dataset.Architects import Architect


if __name__ == '__main__':
    dts = Classification.ClassifactionDataset(
        dir = os.path.join('.', 'test', 'pfli_architect_dataset_test'),
        name = 'myNewDataset',
        architect = Architect.Classification,
    )
    # print(dts)
    for fold, ds_train, ds_test in dts():
        for data in ds_train:
            print(data)