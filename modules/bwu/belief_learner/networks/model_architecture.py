from typing import NamedTuple, Tuple, Optional, List, Union, Callable
import toml
import os


class ModelArchitecture(NamedTuple):
    hidden_units: Optional[Tuple[int, ...]] = None
    activation: Optional[str] = None
    output_dim: Optional[Tuple[int, ...]] = None
    input_dim: Optional[Tuple[int, ...]] = None
    name: Optional[str] = None
    batch_norm: bool = False
    filters: Optional[Tuple[int, ...]] = None
    kernel_size: Optional[Union[Tuple[int, ...], int]] = None
    strides: Optional[Union[Tuple[int, ...], int]] = None
    padding: Union[Tuple[str, ...], str] = None
    raw_last: bool = False
    transpose: bool = False
    invert_model: Optional["ModelArchitecture"] = None

    @property
    def is_cnn(self):
        return self.filters is not None

    def invert(self, original_shape: Optional[Tuple[int, ...]]):
        assert not self.transpose
        model_arch = self._asdict()
        if self.name is not None:
            model_arch['name'] = 'inv_' + self.name
        if original_shape is None:
            original_shape = self.input_dim
        if self.is_cnn:
            model_arch['filters'] = tuple(reversed(self.filters[:-1])) + (original_shape[-1],)
            model_arch['kernel_size'] = tuple(reversed(self.kernel_size))
            model_arch['strides'] = tuple(reversed(self.strides))
            model_arch['padding'] = tuple(reversed(self.padding))
        else:
            model_arch["hidden_units"] = tuple(reversed(self.hidden_units))
        model_arch['output_dim'] = original_shape
        model_arch['input_dim'] = self.output_dim
        model_arch['transpose'] = True

        if self.invert_model is not None:
            del model_arch['invert_model']
            model_arch.update(self.invert_model.short_dict())

        model_arch = ModelArchitecture(**model_arch)
        return model_arch

    def short_dict(self):
        return {k: v for k, v in self._asdict().items() if v is not None}

    @classmethod
    def read_from_toml(cls, file_name):
        with open(file_name, 'r') as f:
            d = toml.load(f)
        return ModelArchitecture(**{key: value if not isinstance(value, list) else tuple(value)
                                    for key, value in d.items()})

    def to_toml(self, file_name):
        assert not os.path.exists(file_name), "File already exists."
        with open(file_name, 'w') as f:
            toml.dump(self.short_dict(), f)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.short_dict())[1:-1]})"

    def replace(self, **kwargs):
        return self._replace(**kwargs)

