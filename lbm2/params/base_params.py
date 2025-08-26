import json
from dataclasses import dataclass, fields

import draccus


@dataclass(frozen=True)
class BaseParams:
    """
    BaseParams is the base class for all parameters. Other params classes inherit from it.
    """

    def __post_init__(self):
        pass

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

    @classmethod
    def from_dict(cls, dict_data):
        # Recursively handle nested BaseParams objects
        processed_dict = {}

        for field_info in fields(cls):
            field_name = field_info.name
            field_type = field_info.type

            if field_name in dict_data:
                field_value = dict_data[field_name]

                if isinstance(field_type, BaseParams) and isinstance(field_value, dict):
                    # Recursively process nested BaseParams
                    processed_dict[field_name] = field_type.from_dict(field_value)
                else:
                    # Use the value as-is
                    processed_dict[field_name] = field_value
            else:
                # Field not in dict_data, will use default
                pass

        # Include any extra fields that aren't in the class definition
        for key, value in dict_data.items():
            if key not in processed_dict:
                processed_dict[key] = value

        cfg_new = draccus.decode(cls, processed_dict)
        return cfg_new


# Make classes that define to_dict method JSON-serializable
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


json._default_encoder = CustomJSONEncoder()
