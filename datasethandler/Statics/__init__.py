from typing import List, Tuple
from datasethandler import Preprocessing

class Static:
    random_state: int = 42
    batch_size: int = 8
    images_size: Tuple[int, int, int] = (224, 224, 3)
    validation: Tuple[bool, float] = (False, 0.2)
    preprocessing: List[str] = [
        Preprocessing.Preproceses.Normlization,
    ]
    class kfold_cross_validation:
        perform: bool = True
        n_split: int = 2
    class train_test_split:
        perform: bool = False
        n_split: int = 1
        test_size: float = 0.2
    def __repr__(self) -> str:
        repr_text = ''
        repr_text += f'Random State: {self.random_state}\n'
        repr_text += f'Batch Size: {self.batch_size}\n'
        repr_text += f'Images Size: {self.images_size}\n'
        repr_text += f'K-Fold Cross Validation: {self.kfold_cross_validation}\n'
        repr_text += f'Validation: {self.validation}\n'
        return repr_text