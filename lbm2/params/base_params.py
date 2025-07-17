from dataclasses import dataclass, field, fields
import draccus

@dataclass(frozen=True)
class BaseParams:
    """
    BaseParams is the base class for all parameters. Other params classes inherit from it.

    It provides a framework for loading parameters from a file through `load_path`.
    Usage: `--model.load_path=some_path.yaml`, `--data.load_path=...`, etc. 
    Can even be something like `--model.unet.load_path=...`.
    
    This argument is useful for recycling presets that we want to use repeatedly.
    Command line arguments still take precedence (i.e., if an overlapping argument is supplied in 
    the command line, it will overwrite the value from the preset yaml). 
    """
    load_path: str = field(default=None)

    def __post_init__(self):
        if self.load_path is not None:
            cfg_new = self.from_file(self.load_path)
            self.from_existing_config(cfg_new, force=False)

    def __iter__(self):
        """Make the class iterable, yielding (field_name, value) pairs."""
        for field_info in fields(self):
            field_name = field_info.name
            field_value = getattr(self, field_name)
            
            # Support nested BaseParams objects
            if isinstance(field_value, BaseParams):
                for nested_name, nested_value in field_value:
                    yield f"{field_name}.{nested_name}", nested_value
            else:
                yield field_name, field_value

    def init_shared_attributes(self, cfg):
        pass

    @classmethod
    def from_file(cls, file_path):
        cfg_new = draccus.load(cls, file_path)
        return cfg_new
        
    def from_existing_config(self, cfg_new, force=False):
        # Looks for fields in cfg_new that are different from the default value and copies them to self
        # If force is True, overwrite even if current value is already set
        # If force is False, only overwrite if current value is equal to the default value
        for field_info in fields(self):
            field_name = field_info.name
            current_value = getattr(self, field_name, None)
            new_value = getattr(cfg_new, field_name, None)
            if isinstance(current_value, BaseParams):
                current_value.from_existing_config(new_value, force=force)
                continue
            default_value = field_info.default
            if new_value != default_value:
                if isinstance(new_value, list) and len(new_value) == 0:
                    continue
                is_default = current_value == default_value or current_value == []
                if not is_default and not force:
                    continue
                print(f"Setting {field_name} to {new_value}")
                object.__setattr__(self, field_name, new_value)
