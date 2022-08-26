import os
import numpy as np
from typing import Dict, List

class Architect:
    class Classification:
        dir_architect: str = 'PF-L_I'
    architect: Classification = Classification.dir_architect
    def extract(self, dir: str) -> np.ndarray:
            if self.architect == self.Classification.dir_architect:
                return self.__pfli(self, dir)
            else:
                raise ValueError(f'Selected Architect Not Valid. {self.architect}')
    def __pfli(self, dir:str) -> np.ndarray:
        list_dataset = []
        for i, label in enumerate(os.listdir(dir)):
            for imagePath in os.listdir(os.path.join(dir, label)):
                list_dataset.append(
                    [
                        os.path.join(dir, label, imagePath),
                        str(i),
                    ]
                )
        list_dataset = np.array(
            list_dataset,
            dtype=str,
        )
        np.random.shuffle(list_dataset)
        return list_dataset
        