from typing import Any, Generator, Tuple
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


from datasethandler.Dataset.Architects import Architect
from datasethandler.Preprocessing import Preproceses
from datasethandler.Statics import Static 


class Dataset:
    def __init__(
        self,
        dir: str,
        name: str ='Dataset',
        architect: Architect = Architect.architect,
    ) -> None:
        self.name = name
        self.dir = dir
        self.architect = architect
        self.list_dataset = self.extract()
    def __call__(self) -> Generator[tuple[int, tf.data.Dataset, tf.data.Dataset], None, None]:
        if Static.kfold_cross_validation.perform:
            for fold, train_ds, test_ds in self.data_kfold():
                train_ds = train_ds.map(
                    lambda inputs: self.processor_train(inputs),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).batch(
                    Static.batch_size,
                    drop_remainder=True
                ).cache().prefetch(tf.data.AUTOTUNE)
                test_ds = test_ds.map(
                    lambda inputs: self.processor_test(inputs),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).batch(
                    Static.batch_size,
                    drop_remainder=True
                ).cache().prefetch(tf.data.AUTOTUNE)
                yield fold, train_ds, test_ds
        elif Static.train_test_split.perform:
            for fold, train_ds, test_ds in self.data_split():
                train_ds = train_ds.map(
                    lambda inputs: self.processor_train(inputs),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).batch(
                    Static.batch_size,
                    drop_remainder=True
                ).cache().prefetch(tf.data.AUTOTUNE)
                test_ds = test_ds.map(
                    lambda inputs: self.processor_test(inputs),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).batch(
                    Static.batch_size,
                    drop_remainder=True
                ).cache().prefetch(tf.data.AUTOTUNE)
                yield fold, train_ds, test_ds
    def __repr__(self) -> str:
        repr_text = ''
        for item in self.__dict__:
            repr_text += f'{item}: {self.__dict__[item]}\n'
        return repr_text
    def extract(self) -> np.ndarray:
        return Architect.extract(Architect, dir=self.dir)
    def imread(self, path: tf.Tensor) -> tf.Tensor:
        file = tf.io.read_file(path)
        image = tf.image.decode_png(file, Static.images_size[-1])
        image = tf.image.resize(image, Static.images_size[:-1], method='nearest')
        image = Preproceses()(image)
        return image
    def processor_image(self, image_path):
        return self.imread(image_path)
    def processor_label(self, label):
        return int(label)
    def processor_train(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_path = inputs[0]
        image = self.processor_image(image_path)
        label = inputs[1]
        label = self.processor_label(int(label))
        return image, label
    def processor_test(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_path = inputs[0]
        image = self.processor_image(image_path)
        label = inputs[1]
        label = self.processor_label(int(label))
        return image, label
    def data_kfold(self) -> Generator[tuple[int, tf.data.Dataset, tf.data.Dataset], None, None]:
        skf = StratifiedKFold(
            Static.kfold_cross_validation.n_split,
            shuffle=True,
            random_state = Static.random_state,
        )
        for fold, (train_idxs, test_idxs) in enumerate(
            skf.split(self.list_dataset[:, 0], 
            self.list_dataset[:, 1])
        ):
            yield fold, \
                tf.data.Dataset.from_tensor_slices(self.list_dataset[train_idxs]), \
                    tf.data.Dataset.from_tensor_slices(self.list_dataset[test_idxs])
    def data_split(self) -> Generator[tuple[int, tf.data.Dataset, tf.data.Dataset], None, None]:
        skf = StratifiedShuffleSplit(
            Static.train_test_split.n_split,
            random_state = Static.random_state,
        )
        for fold, (train_idxs, test_idxs) in enumerate(
            skf.split(self.list_dataset[:, 0], 
            self.list_dataset[:, 1])
        ):
            yield fold, \
                tf.data.Dataset.from_tensor_slices(self.list_dataset[train_idxs]), \
                    tf.data.Dataset.from_tensor_slices(self.list_dataset[test_idxs])