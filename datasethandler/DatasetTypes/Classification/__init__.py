import tensorflow as tf
import numpy as np

from datasethandler.Dataset import Dataset
from datasethandler.Dataset.Architects import Architect


class ClassifactionDataset(Dataset):
    def __init__(
        self,
        dir: str,
        name: str = 'Dataset',
        architect: Architect = Architect.architect
    ) -> None:
        super().__init__(dir, name, architect)
        self.nc = len(np.unique(self.list_dataset[:, 1]))
    def onehot_encoding(self, label: tf.Tensor) -> tf.Tensor:
        onehot_label = self.nc*[0]
        for i in range(self.nc):
            if i == label:
                onehot_label[i] = 1
        return onehot_label
    def processor_label(self, label):
        return self.onehot_encoding(label)