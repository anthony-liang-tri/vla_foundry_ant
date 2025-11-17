import logging
from dataclasses import fields, is_dataclass
from typing import Any, Type, get_origin
from typing import Sequence as SequenceType

from draccus.choice_types import CHOICE_TYPE_KEY
from draccus.parsers.decoding import decode_choice_class
from draccus.utils import is_choice_type


def _strip_unknown_keys(raw_value: Any, cls: Type[Any], path: SequenceType[str]) -> Any:
    """Ignore keys that are not in the dataclass definition while decoding."""
    target_cls = _resolve_dataclass(cls)
    if target_cls is None or not isinstance(raw_value, dict):
        return raw_value

    # For Choice types, determine the actual subclass from the 'type' field
    if is_choice_type(target_cls) and CHOICE_TYPE_KEY in raw_value:
        choice_type = raw_value.get(CHOICE_TYPE_KEY)
        try:
            actual_cls = target_cls.get_choice_class(choice_type)
            # Use the actual subclass for field validation
            target_cls = actual_cls
        except (KeyError, AttributeError):
            # If we can't resolve, use the base class
            pass

    allowed_fields = {field.name for field in fields(target_cls)}
    # Always allow the CHOICE_TYPE_KEY through
    allowed_fields.add(CHOICE_TYPE_KEY)

    cleaned = {key: value for key, value in raw_value.items() if key in allowed_fields}
    removed_fields = [key for key in raw_value if key not in allowed_fields]
    if removed_fields:
        readable_path = ".".join(path) if path else target_cls.__name__
        logging.warning(
            f"Ignoring unknown config fields {', '.join(sorted(removed_fields))} while decoding {readable_path}."
        )

    # Recursively clean nested dataclass fields
    for field_info in fields(target_cls):
        field_name = field_info.name
        if field_name in cleaned and isinstance(cleaned[field_name], dict):
            field_type = field_info.type
            # Resolve the actual type (handle Optional, Union, etc.)
            field_cls = _resolve_dataclass(field_type)
            if field_cls is not None:
                # Recursively strip unknown keys from nested dataclass
                cleaned[field_name] = _strip_unknown_keys(cleaned[field_name], field_type, (*path, field_name))

    return cleaned


def _resolve_dataclass(cls: Type[Any]) -> Type[Any] | None:
    origin = get_origin(cls)
    target_cls = cls if origin is None else origin

    if isinstance(target_cls, type) and is_dataclass(target_cls):
        return target_cls
    return None


def _decode_choice_base_params(cls: Type["BaseParams"], raw_value: Any, path: SequenceType[str]):  # noqa: F821 BaseParams would be circular if imported
    """Handle BaseParams that also behave as Choice types (e.g., DataParams)."""
    if not isinstance(raw_value, dict):
        return decode_choice_class(cls, raw_value, path)

    choice_type = raw_value.get(CHOICE_TYPE_KEY, cls.default_choice_name())
    if choice_type is None:
        return decode_choice_class(cls, raw_value, path)

    try:
        subcls = cls.get_choice_class(choice_type)
    except KeyError:
        return decode_choice_class(cls, raw_value, path)

    payload = {k: v for k, v in raw_value.items() if k != CHOICE_TYPE_KEY}
    cleaned_payload = _strip_unknown_keys(payload, subcls, (*path, choice_type))
    cleaned_payload = cleaned_payload.copy()
    cleaned_payload[CHOICE_TYPE_KEY] = choice_type
    return decode_choice_class(cls, cleaned_payload, path)
