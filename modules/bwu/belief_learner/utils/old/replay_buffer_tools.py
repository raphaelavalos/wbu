from abc import ABC, abstractmethod
import tensorflow as tf
import os


class PriorityHandler(ABC):

    def __init__(self):
        self.max_priority = None

    @abstractmethod
    def update_priority(self, key, value, **kwargs):
        return NotImplemented

    @abstractmethod
    def load_or_initialize_checkpoint(self, dir_path: str):
        return NotImplemented

    @abstractmethod
    def checkpoint(self, *args, **kwargs):
        return NotImplemented


class LossPriority(PriorityHandler):

    def __init__(self, replay_buffer, max_priority: tf.float64 = 10.):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.max_priority = tf.constant(value=max_priority, dtype=tf.float64, name='max_priority')
        self.max_loss = tf.keras.metrics.Mean(name='max_loss', dtype=tf.float64)
        self.min_loss = tf.keras.metrics.Mean(name='min_loss', dtype=tf.float64)

        self._checkpointer = None
        self._manager = None

    def update_priority(self, keys: tf.Tensor, loss: tf.Tensor, **kwargs):
        loss = tf.cast(loss, tf.float64)

        self.max_loss(tf.reduce_max(loss))
        self.min_loss(tf.reduce_min(loss))

        x0 = (self.max_loss.result() - self.min_loss.result()) / 2.
        k = self.max_priority / (self.max_loss.result() - self.min_loss.result())
        priorities = self.max_priority * tf.sigmoid(k * (loss - x0))

        if tf.reduce_any(tf.logical_or(
                tf.math.is_nan(priorities),
                tf.math.is_inf(priorities))):
            tf.print("Priorities with NaN or Inf values detected. Resetting priority handler.")
            priorities = self.max_priority / 2 * tf.ones(shape=tf.shape(keys), dtype=tf.float64)  # unknown priority
            self.max_loss.reset_states()
            self.min_loss.reset_states()

        self.replay_buffer.update_priorities(keys=keys, priorities=priorities)
        return priorities

    def load_or_initialize_checkpoint(self, dir_path: str):
        checkpoint_path = os.path.join(dir_path, 'loss_priority')
        self._checkpointer = tf.train.Checkpoint(max_loss=self.max_loss, min_loss=self.min_loss)
        self._manager = tf.train.CheckpointManager(
            checkpoint=self._checkpointer, directory=checkpoint_path, max_to_keep=1)
        self._checkpointer.restore(self._manager.latest_checkpoint)

    def checkpoint(self, *args, **kwargs):
        assert self._checkpointer is not None and self._manager is not None
        self._manager.save(*args, **kwargs)


class PriorityBuckets(PriorityHandler):

    def __init__(self, replay_buffer, latent_state_size: int):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        size = 2 ** latent_state_size
        self.latent_state_size = latent_state_size
        self._buckets = tf.Variable(
            initial_value=tf.zeros(shape=(size,), dtype=tf.int32), trainable=False, name='bucket')
        self.max_priority = tf.Variable(initial_value=0., dtype=tf.float64, trainable=False, name='max_priority')

        self._checkpointer = None
        self._manager = None

    def update_priority(self, keys: tf.Tensor, latent_states: tf.Tensor, **kwargs):
        batch_size = tf.shape(latent_states)[0]
        self.step_counter.assign_add(batch_size)

        indices = tf.reduce_sum(latent_states * 2 ** tf.range(self.latent_state_size), axis=-1)

        for index in indices:
            self._buckets[index].assign(self._buckets[index] + 1)
        priorities = tf.map_fn(
            fn=lambda index: self.step_counter / self._buckets[index],
            elems=indices,
            fn_output_signature=tf.float64)
        batch_max_priority = tf.reduce_max(priorities)

        if self.max_priority < batch_max_priority:
            self.max_priority.assign(batch_max_priority)

        self.replay_buffer.update_priorities(keys, priorities)
        return priorities

    def load_or_initialize_checkpoint(self, dir_path: str):
        checkpoint_path = os.path.join(dir_path, 'priority_buckets')
        self._checkpointer = tf.train.Checkpoint(
            buckets=self._buckets,
            step_counter=self.step_counter,
            max_priority=self.max_priority)
        self._manager = tf.train.CheckpointManager(
            checkpoint=self._checkpointer, directory=checkpoint_path, max_to_keep=1)
        self._checkpointer.restore(self._manager.latest_checkpoint)

    def checkpoint(self, *args, **kwargs):
        assert self._checkpointer is not None and self._manager is not None
        self._manager.save(*args, **kwargs)
