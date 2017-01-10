from __future__ import absolute_import

import keras
import keras.backend as K

from .models import Model as GandlfModel


class AdaptiveLearningRate(keras.callbacks.Callback):
    """Adapts the learning rate according to the model's loss.

    On each batch, the learning rate for the generator and discriminator are
    adapted according to the loss values of each.

    batch_dis_lr = discriminator_lr * (discriminator_loss / total_loss)
    batch_gen_lr = generator_lr * (generator_loss / total_loss)

    Args:
        discriminator_lr: the maximum discriminator learning rate.
        generator_lr: the maximum generator learning rate.
    """

    def __init__(self, discriminator_lr, generator_lr):
        self.discriminator_lr = discriminator_lr
        self.generator_lr = generator_lr

    def on_batch_end(self, epoch, logs={}):
        if not isinstance(self.model, GandlfModel):
            raise ValueError('The AdaptiveLearningRate callback only works '
                             'for Gandlf models.')

        if (not hasattr(self.model.gen_optimizer, 'lr') or
                not hasattr(self.model.dis_optimizer, 'lr')):
            raise ValueError('To use the Adaptive Learning Rate callback, '
                             'both the generator and discriminator optimizers '
                             'must have an "lr" attribute.')

        gen_loss, dis_loss = 0., 0.
        for key, val in logs.items():
            if key.endswith('gen_loss'):
                if val < 0:
                    raise ValueError('The adaptive learning rate callback '
                                     'doesn\'t work for negative losses.')
                gen_loss += val
            elif key.endswith('real_loss') or key.endswith('fake_loss'):
                if val < 0:
                    raise ValueError('The adaptive learning rate callback '
                                     'doesn\'t work for negative losses.')
                dis_loss += val

        dis_loss /= 2  # Double-counting real and fake data.
        total_loss = gen_loss + dis_loss + 1e-12
        gen_pct, dis_pct = gen_loss / total_loss, dis_loss / total_loss

        # Calculates the percentage to weight each one.
        generator_lr = self.generator_lr * gen_pct
        discriminator_lr = self.discriminator_lr * dis_pct

        # Updates the learning rates on both.
        K.set_value(self.model.gen_optimizer.lr, generator_lr)
        K.set_value(self.model.dis_optimizer.lr, discriminator_lr)
