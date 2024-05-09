import os
from typing import Optional, Callable, Union, Collection, Dict

import imageio
import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing.types import Bool

from PIL import Image, ImageFont, ImageDraw


class VideoEmbeddingObserver:
    def __init__(
            self,
            py_env: PyEnvironment,
            file_name: str,
            fps: int = 30,
            num_episodes: int = 1,
            labeling_function: Optional[Callable[[TimeStep], Union[Collection, Dict[str, Collection]]]] = None,
            font_color: str = 'black',
    ):
        self.py_env = py_env
        self._file_name = file_name
        self.writer = None
        self.fps = fps
        self.best_rewards = -1. * np.inf
        self.cumulative_rewards = 0.
        self.num_episodes = num_episodes
        self.current_episode = 1
        if len(file_name.split(os.path.sep)) > 1 \
                and not os.path.exists(os.path.sep.join(file_name.split(os.path.sep)[:-1])):
                os.makedirs(os.path.sep.join(file_name.split(os.path.sep)[:-1]))
        self.file_name = None
        self.labeling_fn = labeling_function
        self.font_color = font_color

    def __call__(self, time_step: TimeStep, *args, **kwargs):
        if self.writer is None:
            self.writer = imageio.get_writer('{}.mp4'.format(self._file_name), fps=self.fps)
        data = self.py_env.render(mode='rgb_array')
        if self.labeling_fn is not None:
            label = self.labeling_fn(time_step)
            if type(label) is dict:
                label = '\n'.join([str(key)+': '+str(value) for key, value in label.items()])
            else:
                label = str(label)
            img = Image.fromarray(data)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype('Arial', 10)
            except Exception:
                font = ImageFont.load_default()
            if self.font_color == 'black':
                _fill = (0, 0, 0)
            else:
                _fill = (255, 255, 255)
            draw.text((0, 0), label, font=font, fill=_fill)
            data = np.array(img)
        if data is not None:
            self.writer.append_data(data)
        self.cumulative_rewards += time_step.reward
        if time_step.is_last() and self.current_episode < self.num_episodes:
            self.current_episode += 1
        elif time_step.is_last():
            self.writer.close()
            self.writer = None
            avg_rewards = np.sum(self.cumulative_rewards / self.num_episodes)
            if avg_rewards >= self.best_rewards:
                self.best_rewards = avg_rewards
                os.rename('{}.mp4'.format(self._file_name),
                          '{}_rewards={:.2f}.mp4'.format(self._file_name, self.best_rewards))
                self.file_name = '{}_rewards={:.2f}.mp4'.format(self._file_name, self.best_rewards)
            else:
                os.remove('{}.mp4'.format(self._file_name))
            self.cumulative_rewards = 0.
            self.current_episode = 1
