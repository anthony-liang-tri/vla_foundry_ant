# Adding a new model

## 1. Params class

You will need to add a new class to [params/model_params.py](/vla_foundry/params/model_params.py). This class should inherit from `ModelParams`.

At the top of the new class, you need to add the decorator `@register_model_params("insert-model-name")`. This will allow draccus to automatically route to that class with `--model.type`. 

```python
@register_model_params("my_model")
@dataclass(frozen=True)
class MyModelParams(ModelParams):
```

Experiment params are defined here. Any parameter or sub-parameter can be accessed from it in the init_shared_attributes function.

Next, you will need to define the following functions in the params class:
- Initialization: This is where you define all the model parameters needed to initialize your model (e.g., hidden size, number of layers, etc.) If your model is composed of many modular components, you can also define other `ModelParams` objects as attributes below. For instance, in the example below, the class initializes Transformer params and CLIP params as attributes. These can be parsed by draccus using `--model.transformer` and `--model.clip`.

```python
@register_model_params("my_model")
@dataclass(frozen=True)
class MyModelParams(ModelParams):
    my_simple_int: int = field(default=1)
    transformer: Union[TransformerParams, TransformerHFParams] = field(default_factory=ModelParams)
    clip: Union[CLIPHFParams, CLIP_OpenCLIPParams] = field(default_factory=CLIPHFParams)
    my_shared_param: int = field(default=None)
    my_half_int: float = field(default=None)
```

- `post_init()`: This is called immediately after the object is initialized. This is where you can do things like handle null arguments or add assert statements for invalid arguments.

```python
@register_model_params("my_model")
@dataclass(frozen=True)
class MyModelParams(ModelParams):
    my_simple_int: int = field(default=1)
    transformer: Union[TransformerParams, TransformerHFParams] = field(default_factory=ModelParams)
    clip: Union[CLIPHFParams, CLIP_OpenCLIPParams] = field(default_factory=CLIPHFParams)
    my_shared_param: int = field(default=None)
    my_half_int: float = field(default=None)

    def __post_init__(self):
        assert self.my_simple_int > 0, "MyModelParams requires a positive integer value"
        object.__setattr__(self, "my_half_int", float(my_simple_int)/2)
```

- `init_shared_attributes(self, cfg)`: This function takes in the full cfg object and allows you to access parameters which are defined in other params classes. For instance, if `my_shared_param` is owned by `DataParams` but we want to access it in `MyModelParams`, then this is where it needs to be defined. 

Note that `my_shared_param` still needs to exist as an attribute in `MyModelParams`, but you should not be passing `--model.my_shared_param` and instead passing `--data.my_shared_param`, then letting the `init_shared_attributes()` function handle the param copying. Essentially draccus first loads all the params, then recursively calls `init_shared_attributes()` during the `__post_init__` stage. During these `init_shared_attributes()` calls, the shared value is copied from one class to another, in the process overwriting the existing value.

Params are frozen and immutable. Here, we deliberately override this with `object.__setattr__()`.

```python
@register_model_params("my_model")
@dataclass(frozen=True)
class MyModelParams(ModelParams):
    my_simple_int: int = field(default=1)
    transformer: Union[TransformerParams, TransformerHFParams] = field(default_factory=ModelParams)
    clip: Union[CLIPHFParams, CLIP_OpenCLIPParams] = field(default_factory=CLIPHFParams)
    my_shared_param: int = field(default=None)
    my_half_int: float = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "my_half_int", float(my_simple_int)/2)

    def init_shared_attributes(self, cfg):
    """
	Called at the initialization of training with cfg the experiment params.
    """
        super().init_shared_attributes(cfg)
        object.__setattr__(self, "my_shared_param", cfg.data.my_shared_param)
```

## 2. Model definition

Using these parameters, you can define your model in a new file under [vla_foundry/models/](/vla_foundry/models/)

```python
from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.transformer import Transformer
from vla_foundry.models.transformer_hf import TransformerHF
from vla_foundry.params.model_params import MyModelParams

class MyModel(BaseModel):
    def __init__(self, model_params: MyModelParams, transformer: Union[Transformer, TransformerHF]):
        super().__init__()
	 self.transformer = transformer

    def forward(
        self,
        input_ids,
        inputs_embeds,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        output_hidden_states=False,
        is_causal=None,
    ):
        """
        Args: The same args as the Transformer model
        """

	# This is a simple example but let's say we just want a transformer that divides the inputs_embeds by the global batch size
	inputs_embeds = inputs_embeds/self.model_params.my_shared_param
	return self.transformer(input_ids, inputs_embeds, past_key_values, use_cache, attention_mask, output_hidden_states, is_causal)
```

### 3. Add model to registry

Next, you will define your model in the `create_model()` function of [models/\_\_init\_\_.py](/vla_foundry/models/__init__.py).

```python
    elif model_params.type == "my_model":
        transformer = create_model(model_params.transformer)
        model = MyModel(model_params, transformer)
```

If you wish to use FSDP, you will also need to add your model to the `get_model_block()` registry in [models/\_\_init\_\_.py](/vla_foundry/models/__init__.py). This usually requires extracting out the "block" unit of your model for FSDP to wrap around. See example below.

```python
    if model_type == "transformer":
        return (TransformerBlock,)
    elif model_type == "my_model":
        transformer_block = get_model_block(model_params.transformer.type, model_params.transformer)
        clip_block = get_model_block(model_params.clip.type, model_params.clip)
        return (*transformer_block, *clip_block)
```
