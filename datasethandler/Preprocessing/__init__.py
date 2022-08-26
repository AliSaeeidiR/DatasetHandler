import tensorflow as tf

class Preproceses:
    Normlization = 'MinMax-Normalization'
    def __call__(self, images):
        if self.Normlization == 'MinMax-Normalization':
            return self._normalize(images)
        else:
            raise ValueError(f'Selected Normalization Is Not Valid. {self.Normlization}')
    def _normalize(self, image: tf.Tensor):
        return tf.divide(
                tf.subtract(
                    image, 
                    tf.reduce_min(image)
                ), 
                tf.subtract(
                    tf.reduce_max(image), 
                    tf.reduce_min(image)
                )
            )