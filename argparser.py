"""
Modified from HfArgumentParser
"""
import dataclasses
import json
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from dataclasses import asdict
from enum import Enum
from inspect import isclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, NewType, Optional, Tuple, Union, get_type_hints

import yaml

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


@dataclasses.dataclass
class _BaseArgumentDataClass:
    def __post_init__(self):
        pass


def _is_argclass(obj):
    return issubclass(obj, _BaseArgumentDataClass) if isinstance(obj, type) else _is_argclass(type(obj))


def _flatten_args(json):
    if type(json) == dict:
        for k, v in list(json.items()):
            if type(v) == dict:
                _flatten_args(v)
                json.pop(k)
                for k2, v2 in v.items():
                    json[k + "." + k2] = v2


def _unflatten_args(json):
    if type(json) == dict:
        for k in sorted(json.keys(), reverse=True):
            if "." in k:
                key_parts = k.split(".")
                json1 = json
                for i in range(0, len(key_parts) - 1):
                    k1 = key_parts[i]
                    if k1 in json1:
                        json1 = json1[k1]
                        if type(json1) != dict:
                            conflicting_key = ".".join(key_parts[0:i + 1])
                            raise Exception('Key "{}" conflicts with key "{}"'.format(
                                k, conflicting_key))
                    else:
                        json2 = dict()
                        json1[k1] = json2
                        json1 = json2
                if type(json1) == dict:
                    v = json.pop(k)
                    json1[key_parts[-1]] = v


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def _string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def argclass(*args, **kwargs):
    """
    wrapper to create arg class
    """

    def decorator(cls):
        should_add_arg_base = True
        for base in cls.__bases__:
            if issubclass(base, _BaseArgumentDataClass):
                should_add_arg_base = False
        if should_add_arg_base:
            cls = type(cls.__name__, (_BaseArgumentDataClass, *cls.__bases__), dict(cls.__dict__))
        # update argument annotations if there are any
        for name, field_type in cls.__annotations__.items():
            if _is_argclass(field_type) and getattr(cls, name, None) is None:
                setattr(cls, name, field_type())

        # decode dictionaries into argument classes through post init
        original_post_init = getattr(cls, '__post_init__', None)

        def __post_init__(self):
            for name, field_type in cls.__annotations__.items():
                field_value = getattr(self, name, None)
                if _is_argclass(field_type) and type(field_value) is dict:
                    setattr(self, name, field_type(**field_value))

            if original_post_init is not None and callable(original_post_init):
                original_post_init(self)

        cls.__post_init__ = __post_init__

        cls = dataclasses.dataclass(cls, **kwargs)
        return cls

    return decorator(args[0]) if args else decorator


# Modified from HfArgumentParser
class NestedArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.
    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace. Optional: To create sub argument groups use the `_argument_group_name` attribute in the dataclass.
    """

    dataclass_type: DataClassType

    def __init__(self, dataclass_type: DataClassType, required_args=None, **kwargs):
        """
        Args:
            dataclass_type:
                Dataclass type
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        # To make the default appear when using --help
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        self.dataclass_type = dataclass_type
        self.required_args = required_args
        if self.required_args is None:
            self.required_args = []

        self._add_dataclass_arguments(self.dataclass_type)

    def _parse_dataclass_field(self, parser: ArgumentParser, field: dataclasses.Field, parent=None):
        # fold parent name into field for nested arguments
        if parent is None:
            parent = ''
        else:
            parent = f'{parent}.'

        # ignore arguments with _ in beginning
        if field.name.startswith('_'):
            return
        field_name = f"--{parent}{field.name}"
        kwargs = field.metadata.copy()
        # field.metadata is not used at all by Data Classes,
        # it is provided as a third-party extension mechanism.
        if isinstance(field.type, str):
            raise RuntimeError(
                "Unresolved type detected, which should have been done with the help of "
                "`typing.get_type_hints` method by default"
            )

        origin_type = getattr(field.type, "__origin__", field.type)
        if origin_type is Union:
            if str not in field.type.__args__ and (
                    len(field.type.__args__) != 2 or type(None) not in field.type.__args__
            ):
                raise ValueError(
                    "Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because"
                    " the argument parser only supports one type per argument."
                    f" Problem encountered in field '{field.name}'."
                )
            if type(None) not in field.type.__args__:
                # filter `str` in Union
                field.type = field.type.__args__[0] if field.type.__args__[1] == str else field.type.__args__[1]
                origin_type = getattr(field.type, "__origin__", field.type)
            elif bool not in field.type.__args__:
                # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
                field.type = (
                    field.type.__args__[0] if isinstance(None, field.type.__args__[1]) else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)

        # A variable to store kwargs for a boolean field, if needed
        # so that we can init a `no_*` complement argument (see below)
        bool_kwargs = {}
        if isinstance(field.type, type) and issubclass(field.type, Enum):
            kwargs["choices"] = [x.value for x in field.type]
            kwargs["type"] = type(kwargs["choices"][0])
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
        elif dataclasses.is_dataclass(field.type):
            self._add_dataclass_arguments(field.type, parent=f'{parent}{field.name}')
            return
        elif field.type is bool or field.type == Optional[bool]:
            # Copy the currect kwargs to use to instantiate a `no_*` complement argument below.
            # We do not initialize it here because the `no_*` alternative must be instantiated after the real argument
            bool_kwargs = copy(kwargs)

            # Hack because type=bool in argparse does not behave as we want.
            kwargs["type"] = _string_to_bool
            if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                # Default value is False if we have no default when of type bool.
                default = False if field.default is dataclasses.MISSING else field.default
                # This is the value that will get picked if we don't include --field_name in any way
                kwargs["default"] = default
                # This tells argparse we accept 0 or 1 value after --field_name
                kwargs["nargs"] = "?"
                # This is the value that will get picked if we do --field_name (without value)
                kwargs["const"] = True
        elif isclass(origin_type) and issubclass(origin_type, list):
            kwargs["type"] = field.type.__args__[0]
            kwargs["nargs"] = "+"
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        else:
            kwargs["type"] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True

        if field_name in self.required_args:
            kwargs['required'] = True
        parser.add_argument(field_name, **kwargs)

        # Add a complement `no_*` argument for a boolean field AFTER the initial field has already been added.
        # Order is important for arguments with the same destination!
        # We use a copy of earlier kwargs because the original kwargs have changed a lot before reaching down
        # here and we do not need those changes/additional keys.
        if field.default is True and (field.type is bool or field.type == Optional[bool]):
            bool_kwargs["default"] = False
            parser.add_argument(f"--{parent}no_{field.name}", action="store_false", dest=f"{parent}{field.name}",
                                **bool_kwargs)

    def _add_dataclass_arguments(self, dtype: DataClassType, parent=None):
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self

        try:
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            raise RuntimeError(
                f"Type resolution failed for f{dtype}. Try declaring the class in global scope or "
                "removing line of `from __future__ import annotations` which opts in Postponed "
                "Evaluation of Annotations (PEP 563)"
            )

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field.type = type_hints[field.name]
            self._parse_dataclass_field(parser, field, parent=parent)

    def parse_args(self, args=None, return_entered_args=False) -> DataClass:
        namespace, remaining_args = self.parse_known_args(args=args)

        if len(remaining_args) > 0:
            pprint("Didn't recognize the following arguments:")
            pprint(remaining_args)
            exit(1)
        default_args = asdict(self.dataclass_type())
        _flatten_args(default_args)
        inputs = {k: v for k, v in vars(namespace).items()}
        entered_args = [arg.replace('--', '').split('=')[0] for arg in sys.argv]
        entered_args = [arg for arg in entered_args if arg in default_args.keys()]
        _unflatten_args(inputs)
        obj = self.dataclass_type(**inputs)

        if return_entered_args:
            return obj, entered_args
        else:
            return obj

    def parse_json_file(self, json_file: str) -> DataClass:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        keys = {f.name for f in dataclasses.fields(self.dataclass_type) if f.init}
        inputs = {k: v for k, v in data.items() if k in keys}
        _unflatten_args(inputs)
        obj = self.dataclass_type(**inputs)
        return obj

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.
        """
        keys = {f.name for f in dataclasses.fields(self.dataclass_type) if f.init}
        inputs = {k: v for k, v in args.items() if k in keys}
        _unflatten_args(inputs)
        obj = self.dataclass_type(**inputs)
        return obj


def parse_config(config_file):
    assert os.path.exists(config_file), f'could not find config file: {config_file}'

    if '.yaml' in config_file:
        with open(config_file) as f:
            arg_dict = (yaml.safe_load(f))
    elif '.json' in config_file:
        with open(config_file) as f:
            arg_dict = json.load(f)
    else:
        assert False, f'{config_file} file type is not supported yet!'
    return arg_dict


def parse_args(arg_class, required_args=None, print_args=True, resolve_config=True):
    """
    behavior as follows: first reads config file arguments. then overrides any arguments with
     provided command line arguments
    @param print_args: print parsed arguments into
    @param required_args: list of arguments that must be passed; assertion error if not passed
    @param arg_class: @argclass wrapped class to populate
    @param resolve_config: should automatically read and parse config file
    :return:
    """

    def resolve_config_file(config):
        # if config folder provided instead of config, use file name to find config.yaml file
        if os.path.isdir(config):
            script = sys.argv[0]
            script = os.path.basename(script)
            script = script.replace('.py', '.yaml')
            config = os.path.join(config, script)
        return config

    parser = NestedArgumentParser(arg_class, required_args=required_args)
    cli_args, entered_args = parser.parse_args(return_entered_args=True)
    cli_arg_dict = asdict(cli_args)
    _flatten_args(cli_arg_dict)
    cli_arg_dict = {arg: cli_arg_dict[arg] for arg in entered_args}

    # check if config file is provided
    arg_dict = {}

    if 'config' in cli_arg_dict.keys() and cli_arg_dict['config'] and resolve_config:
        cli_arg_dict['config'] = resolve_config_file(cli_arg_dict['config'])
        config_file = cli_arg_dict['config']
        arg_dict = parse_config(config_file)

    # override config file args with command line args
    _flatten_args(arg_dict)
    arg_dict.update(cli_arg_dict)

    _unflatten_args(arg_dict)
    # parse into dataclass representation
    args = arg_class(**arg_dict)

    if ('no_print_args' in arg_dict and arg_dict['no_print_args']) or print_args:
        pprint("=" * 20 + " Training Arguments " + "=" * 20)
        pprint(asdict(args))
    return args
