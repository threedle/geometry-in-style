"""
Threedle configs: Thronf
"""

"""
MIT License

Copyright (c) 2024 namanh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# namanh: I started this before I knew pyrallis existed... this does very similar things, in
# just one file, no packages :-)
# requires python >=3.8

from typing import TypeVar, Type, Callable, Any, Union, Sequence, Tuple, Mapping
from typing import Optional, get_args, get_origin, cast
from typing import Dict, TextIO, Literal, List
import collections.abc as abc
from functools import partial, wraps
from dataclasses import dataclass, fields, asdict, MISSING, replace, Field, _MISSING_TYPE
import os
import sys
import json
import argparse
from copy import deepcopy  # only to copy an argparser


_T = TypeVar("_T")
ThronfigSubclassT = TypeVar("ThronfigSubclassT", bound="Thronfig")
ATOMIC_TYPES = (str, bool, int, float, type(None))
ThronfigValueT = Union[
    str, bool, int, float, type(None), "Thronfig", Sequence["ThronfigValueT"]
]
PreInitValueT = Union[
    str, bool, int, float, type(None), "_PreInitT", Sequence["PreInitValueT"]
]
_PreInitT = Mapping[str, PreInitValueT]
""" values of a complex object thronfig field before it is properly init'd """


class MISSING_pretty_type:
    def __repr__(self):
        return "<MISSING>"


# to get default field values from dataclasses with slots=True
_member_descriptor_type = type(type("A", (), {"__slots__": ("A")}).A)  # type: ignore

MISSING_pretty = MISSING_pretty_type()

DEFAULT_IGNORE_UNKNOWN_FIELDS = True


class InvalidConfigError(TypeError):
    """user-raised error for missing config expectations not specified in the type"""


class ConfigDoesNotTypecheck(TypeError):
    """lib-raised error for configs that fail to meet their declared types"""


class ConfigArgumentPatchingError(argparse.ArgumentTypeError):
    """when argparse argpatching into Thronfigs fails"""


def _only_expected_kwargs(
    f: Callable[..., Any], kwargs_dict: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    given a function f and a dict to be used like f(**kwargs_dict), filter
    the kwargs_dict to contain only valid arg names for f
    """
    return {k: kwargs_dict[k] for k in f.__code__.co_varnames if (k in kwargs_dict)}


def _all_expected_kwargs(
    f: Callable[..., Any],
    kwargs_dict: Mapping[str, Any],
    default_value: Any = MISSING,
    exclude_varnames: Sequence[str] = ("self", "cls"),
) -> Dict[str, Any]:
    """not used, but this is how you'd init a fully empty dataclass"""
    return {
        k: (kwargs_dict[k] if k in kwargs_dict else default_value)
        for k in f.__code__.co_varnames
        if (k not in exclude_varnames)
    }


def _patch_dict_inplace(
    orig_dict: Dict[str, Any],
    patch_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    for patch_key, patch_val in patch_dict.items():
        if isinstance(patch_val, _MISSING_TYPE):
            continue
        patch_val_is_dict = isinstance(patch_val, Mapping)
        if (orig_val := orig_dict.get(patch_key)) is None:
            # field not found; if the patch val is a dict, make a new complex
            # field (empty dict or appropriate empty Thronfig)
            if patch_val_is_dict:
                orig_val = dict()
                orig_dict[patch_key] = orig_val
        if isinstance(orig_val, dict):
            assert patch_val_is_dict, (
                f"cannot patch object at key {patch_key} with non-object value {patch_val}!"
            )
            _patch_dict_inplace(orig_val, patch_val)
        else:
            orig_dict[patch_key] = patch_val
    return orig_dict


def _open_fname_or_file(
    fname_or_file: Union[str, os.PathLike, TextIO, None],
    open_mode: Literal["r", "w"],
    treat_hyphen_path_as_stdin: bool = False,
):
    # the return of this goes straight into a with statement, which can't handle None, so we
    # just raise a FileNotFoundError hoping that whatever is catching exceptions from this
    # function will also catch FileNotFoundError
    if fname_or_file is None:
        raise FileNotFoundError(f"cannot open fname or file given fname_or_file=None")
    if (
        treat_hyphen_path_as_stdin
        and isinstance(fname_or_file, (str, os.PathLike))
        and fname_or_file == "-"
    ):
        fname_or_file = sys.stdin
        assert fname_or_file is not None

    return (
        open(fname_or_file, open_mode)
        if isinstance(fname_or_file, (str, os.PathLike))
        else fname_or_file
    )


class _ThronfigArgparseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """namespace will only contain one item: the patch_dict"""
        assert option_string is not None
        return _ThronfigArgparseAction.read_in_option_strings(
            namespace, values, option_string
        )

    @staticmethod
    def read_in_option_strings(namespace: argparse.Namespace, values, option_string: str):
        """patches namespace._patch_dict field specified by option_string with values"""
        assert option_string.startswith("--")
        accesses = option_string[2:].split(".")
        #  e.g. "--main.io.load_fname 'asdf.txt'" gives accesses = ["main", "io", "load_fname"]

        assert isinstance(namespace._patch_dict, dict)
        curr_parent_of_key_to_patch = namespace._patch_dict
        n_accesses = len(accesses)
        for i, access in enumerate(accesses):
            is_last_access = i == n_accesses - 1
            if access not in curr_parent_of_key_to_patch:
                curr_parent_of_key_to_patch[access] = None if is_last_access else dict()
            child_value = curr_parent_of_key_to_patch[access]
            if isinstance(child_value, dict):
                curr_parent_of_key_to_patch = child_value
                # continue down the stack of dict accesses
                continue
            else:
                # if we reached an atomic value and it is also the last access in the stack
                # of accesses then modify the value in the patch_dict
                if i == n_accesses - 1:
                    curr_parent_of_key_to_patch[access] = values
                    break

            # if none of the branches above terminated or continued the loop,
            raise ConfigArgumentPatchingError(
                f"patching argument {option_string} is not an atomic value that can be patched from the command line"
            )


class _ArgTypeReader:
    """for use as the type reader function in ArgumentParser.add_argument.
    All this could have just been a curried function (where fieldtype and the flag are
    partially applied), but somehow py 3.12 broke all of that and assigned wrong type reader
    functions to the arguments so I have to use an object now"""

    def __init__(self, fieldtype: Type, allow_parsing_none_and_null: bool):
        if fieldtype is str:
            self.reader = str
        else:
            self.reader = json.loads

        self.allow_parsing_none_and_null = allow_parsing_none_and_null

    def __call__(self, s: str):
        if self.allow_parsing_none_and_null:
            if s == "None" or s == "null":
                return None

        try:
            return self.reader(s)
        except json.JSONDecodeError as e:
            if not s:
                return None
            # return the string, hope that any error will be caught at typecheck instead
            if s[0] == "[" and s[-1] == "]":
                # special handling for strings inside JSON arrays: since the shell might eat
                # the double quotes around strings. For instance, --arg ["foo","bar"] leaves
                # the string to pass to json.loads to be the invalid '[foo,bar]' rather than
                # '["foo","bar"]'. Warn the user to try wrapping the array in verbatim quotes ''
                raise argparse.ArgumentTypeError(
                    f'could not parse as JSON array: {s}. Maybe your shell has eaten the string quotes from the string items in the array; try wrapping the array in verbatim quotes like \'["a","b"]\' or escape the quotes like [\\"a\\",\\"b\\"].\nThe JSON decoder gave this error:\n{e.args}'
                ) from e
            return s


def _parse_args_mine(
    parser: argparse.ArgumentParser,
    args: Optional[Sequence[str]],
    expect_all_args_to_be_known: bool,
) -> Tuple[argparse.Namespace, Sequence[str]]:
    init_namespace = argparse.Namespace(_patch_dict=dict())
    if expect_all_args_to_be_known:
        namespace = parser.parse_args(args, namespace=init_namespace)
        remains = []
    else:
        namespace, remains = parser.parse_known_args(args, namespace=init_namespace)
    return namespace, remains


def _format_typename(ty: Type) -> str:
    if get_origin(ty) is not None:
        return repr(ty)
    else:
        return ty.__qualname__


@dataclass
class Thronfig:
    def __replace_shallow(self: ThronfigSubclassT, other: ThronfigSubclassT):
        for fieldspec in fields(self):
            setattr(self, fieldspec.name, getattr(other, fieldspec.name))
        # no need to typecheck, since 'other' would have already been typechecked when it
        # was created (via from_dict)

    def __post_typecheck__(self):
        # to be overridden by superclasses, and to be run recursively (from bottom up)
        pass

    @classmethod
    def __get_default_field_value(
        cls, fieldspec: Field, value_for_nodefaults: Any = MISSING
    ) -> Union[ThronfigValueT, Any]:
        if not isinstance(
            (val := getattr(cls, fieldspec.name, MISSING)),
            (_MISSING_TYPE, _member_descriptor_type),
        ):
            return val
        if not isinstance((val := fieldspec.default), _MISSING_TYPE):
            return val
        if not isinstance((val := fieldspec.default_factory), _MISSING_TYPE):
            return val()
        return value_for_nodefaults

    @classmethod
    def make_default_dict(
        cls: Type[ThronfigSubclassT], value_for_nodefaults: Any = MISSING
    ) -> Dict[str, Any]:
        return {
            fieldspec.name: (
                classval := cls.__get_default_field_value(fieldspec, value_for_nodefaults),
                (
                    classval.__class__.make_default_dict(value_for_nodefaults)
                    if isinstance(classval, Thronfig)
                    else classval
                ),
            )[-1]
            for fieldspec in fields(cls)
        }

    @classmethod
    def make_default(
        cls: Type[ThronfigSubclassT], value_for_nodefaults: Any = MISSING
    ) -> ThronfigSubclassT:
        return cls.from_dict(cls.make_default_dict(value_for_nodefaults))

    @classmethod
    def from_dict(
        cls: Type[ThronfigSubclassT],
        dictionary: _PreInitT,
        patch_dict: Optional[_PreInitT] = None,
        ignore_unknown_fields: bool = DEFAULT_IGNORE_UNKNOWN_FIELDS,
    ) -> ThronfigSubclassT:
        """
        - dictionary: some Mapping-like object to convert into a Thronfig dataclass
            (does not have to contain all fields required by the dataclass if there are
            defaults; this automatically fills missing keys with dataclass-declared defaults
        - patch_dict: (Optional) another Mapping-like whose fields will override the
          corresponding fields in `dictionary`
        - ignore_unknown_fields: if True, will silently ignore field names not
          declared in the Thronfig dataclass declaration(s)
            If False, will throw on unknown field names.
        """
        default_dict = cls.make_default_dict()
        # slap the input dict onto the default dict
        dict_to_convert = dict(
            _only_expected_kwargs(cls.__init__, dictionary)
            if ignore_unknown_fields
            else dictionary
        )
        dict_to_convert = _patch_dict_inplace(default_dict, dict_to_convert)
        # then apply the patch dict too, if specified
        if patch_dict is not None:
            dict_to_convert = _patch_dict_inplace(dict_to_convert, patch_dict)

        # this and typecheck_and_convert are the only places in this module
        # where we init cls using its constructor (everything else will call
        # this from_dict classmethod here)
        thronfig = cls(**_all_expected_kwargs(cls.__init__, dict_to_convert))

        # then we can typecheck
        # thronfig.typecheck_and_convert__bad(ignore_unknown_fields=ignore_unknown_fields)
        thronfig.typecheck_and_convert(ignore_unknown_fields=ignore_unknown_fields)

        return thronfig

    @classmethod
    def from_json_string(
        cls: Type[ThronfigSubclassT],
        json_string: str,
        ignore_unknown_fields: bool = DEFAULT_IGNORE_UNKNOWN_FIELDS,
    ) -> ThronfigSubclassT:
        loaded_dict = json.loads(json_string)
        return cls.from_dict(loaded_dict, ignore_unknown_fields=ignore_unknown_fields)

    @classmethod
    def load_from_json_file(
        cls: Type[ThronfigSubclassT],
        fname_or_file: Union[str, os.PathLike, TextIO],
        ignore_unknown_fields: bool = DEFAULT_IGNORE_UNKNOWN_FIELDS,
        json_patch_fname: Optional[str] = None,
    ) -> ThronfigSubclassT:
        with _open_fname_or_file(fname_or_file, "r", treat_hyphen_path_as_stdin=True) as f:
            loaded_dict = json.load(f)
            if json_patch_fname is not None:
                # the patch file cannot be - (stdin)
                with _open_fname_or_file(json_patch_fname, "r") as pf:
                    _patch_dict_inplace(loaded_dict, json.load(pf))
            return cls.from_dict(loaded_dict, ignore_unknown_fields=ignore_unknown_fields)

    def typecheck_and_convert(
        self, ignore_unknown_fields: bool = DEFAULT_IGNORE_UNKNOWN_FIELDS
    ):
        # better using explicit recursion
        # for formatting Union errors, we keep track of how many indents we're at
        indent = lambda n, s: "\n".join(map(lambda line: " " * n + line, s.split("\n")))

        def go(
            name: str,
            seq_indices: Tuple[int, ...],
            expected_type: Type,
            value: Union[PreInitValueT, ThronfigValueT],
            union_nest_level: int,
            is_directly_union_branch: bool,
        ) -> ThronfigValueT:
            formatted_seq_inds = "".join(("[{}]".format(i)) for i in seq_indices)

            def assert_(cond: Any, badvalue: str = "", badtype: str = ""):
                if not cond:
                    if isinstance(value, Thronfig):
                        for fieldspec in fields(value):
                            if isinstance(getattr(value, fieldspec.name), _MISSING_TYPE):
                                setattr(value, fieldspec.name, MISSING_pretty)
                    raise ConfigDoesNotTypecheck(
                        f"""
  - {self.__class__.__qualname__}{name}{formatted_seq_inds} {"expects type" if not is_directly_union_branch else "can be of type"}
        {_format_typename(expected_type)}"""
                        + (
                            f"""
    but got value
        {repr(value)}
    which {badvalue}"""
                        )
                        if not badtype
                        else f" which is not a valid type as it {badtype}"
                    )

            # seq_indices is current index of the value if it's nested in some sequence parent value
            # explicitly handle the type origins
            origin = get_origin(expected_type)
            argtypes = get_args(expected_type)
            if origin is tuple:
                assert_(
                    isinstance(value, Sequence) and not isinstance(value, str),
                    "is not a tuple",
                )
                # we have this assert as a function call but pyright doesn't realize it has an assert in it, so
                value = cast(Sequence, value)
                assert_(
                    len(value) == len(argtypes),
                    f"does not have the same number of elements as there are types in {expected_type}",
                )
                if Ellipsis in argtypes:
                    assert_(
                        len(argtypes) == 2 and argtypes[0] is not Ellipsis,
                        badtype="has ... in it but not in the only correct form Tuple[T,...]",
                    )

                    argtypes = tuple(argtypes[0] for _ in value)
                return tuple(
                    go(name, seq_indices + (i,), argtype, elem, union_nest_level, False)
                    for i, (argtype, elem) in enumerate(zip(argtypes, value))
                )
            elif origin is abc.Sequence:
                assert_(len(argtypes) == 1, badtype="has more than one type parameter")
                assert_(
                    isinstance(value, Sequence) and not isinstance(value, str),
                    "is not a sequence",
                )
                value = cast(Sequence, value)
                elemtype = argtypes[0]
                return tuple(
                    go(name, seq_indices + (i,), elemtype, elem, union_nest_level, False)
                    for i, elem in enumerate(value)
                )
            elif origin is Union:
                typecheck_errs: List[Union[AssertionError, ConfigDoesNotTypecheck]] = []
                for argtype in argtypes:
                    try:
                        return go(
                            name, seq_indices, argtype, value, union_nest_level + 1, True
                        )
                    except (AssertionError, ConfigDoesNotTypecheck) as typecheck_err:
                        typecheck_errs.append(typecheck_err)
                        continue
                # still here means all Union types have failed to match
                # gather up all the typecheck errors in a big report
                all_reports = "".join(
                    map(
                        partial(indent, 2 * (union_nest_level + 1)),
                        (",".join(err.args) for err in typecheck_errs),
                    )
                )
                assert_(
                    False,
                    "was not of any of the types in the Union, with the following errors when trying each possible type:\n  "
                    + all_reports,
                )
            elif origin is Literal:
                assert_(
                    (valtype := type(value)) in ATOMIC_TYPES,
                    f"is of type {valtype} which cannot be matched with primitive values in type {expected_type}",
                )
                # values in origin should all be primitive hashable types so we can just return them as is
                assert_(value in argtypes, f"is not one of the choices in {expected_type}")
                return cast(ThronfigValueT, value)
            # TODO add cases for origin Mapping or Dict
            elif origin is None:
                # class type, no origin
                if issubclass(expected_type, Thronfig):
                    assert_(
                        isinstance(value, Mapping) or isinstance(value, Thronfig),
                        "is not a dict-like mapping or Thronfig dataclass",
                    )
                    value = cast(Union[Mapping, Thronfig], value)
                    dict_: Mapping[str, Union[PreInitValueT, ThronfigValueT]] = (
                        dict(value) if isinstance(value, Mapping) else value.to_dict()
                    )
                    dict_ = _patch_dict_inplace(expected_type.make_default_dict(), dict_)
                    for fieldspec in fields(expected_type):
                        fieldname = fieldspec.name
                        fieldtype = cast(Type, fieldspec.type)
                        if fieldname in dict_:
                            assert_(
                                not isinstance(dict_[fieldname], _MISSING_TYPE),
                                f"has no specified value for field '{fieldname}' but the {_format_typename(expected_type)} field '{fieldname}' had no default declared either",
                            )
                            dict_[fieldname] = go(
                                name + formatted_seq_inds + "." + fieldname,
                                (),
                                fieldtype,
                                dict_[fieldname],
                                union_nest_level,
                                False,
                            )
                    expected_thronfig = expected_type(
                        **(
                            _only_expected_kwargs(expected_type.__init__, dict_)
                            if ignore_unknown_fields
                            else dict_
                        )
                    )
                    # this is a typechecked and converted thronfig, run __post_typecheck__
                    expected_thronfig.__post_typecheck__()
                    return expected_thronfig
                else:
                    # allow one type of value cast: if the expected type is float and the
                    # value is an int (... or bool, but such is python and its funny bools)
                    if expected_type is float and isinstance(value, int):
                        return float(value)
                    assert_(
                        isinstance(value, expected_type),
                        f"is not a primitive value of type {_format_typename(expected_type)}",
                    )

                    return cast(ThronfigValueT, value)

        # now can run go on self as Thronfig!
        converted = go("", (), self.__class__, self, 0, False)
        assert isinstance(converted, self.__class__)
        self.__replace_shallow(converted)

    @classmethod
    def __register_argparsers_recursively(
        cls,
        parent_argparser_or_group: Union[argparse.ArgumentParser, argparse._ArgumentGroup],
        argparse_prefix: str,
        config_fields_mapped_from_namespace_fields: Optional[Sequence[str]],
        allow_parsing_none_and_null: bool = True,
        args_required_if_no_default: bool = False,
    ):
        for fieldspec in fields(cls):
            fieldname = fieldspec.name
            fieldtype = cast(Type, fieldspec.type)
            if args_required_if_no_default:
                this_arg_is_required = isinstance(
                    cls.__get_default_field_value(fieldspec, value_for_nodefaults=MISSING),
                    _MISSING_TYPE,
                )
            else:
                this_arg_is_required = False

            # strip optional from fieldtype
            found_none_in_union = False
            if get_origin(fieldtype) is Union:
                _args = get_args(fieldtype)
                args_without_none = tuple(arg for arg in _args if arg is not type(None))
                if (_n_args_without_none := len(args_without_none)) == 1:
                    fieldtype = args_without_none[0]
                if _n_args_without_none < len(_args):
                    found_none_in_union = True

            if get_origin(fieldtype) is None and issubclass(fieldtype, Thronfig):
                child_argparser_prefix = argparse_prefix + (
                    fieldname if argparse_prefix == "" else "." + fieldname
                )
                child_argparser_group = parent_argparser_or_group.add_argument_group(
                    child_argparser_prefix
                )
                fieldtype.__register_argparsers_recursively(
                    child_argparser_group,
                    child_argparser_prefix,
                    config_fields_mapped_from_namespace_fields,
                    args_required_if_no_default=(
                        args_required_if_no_default if this_arg_is_required else False
                    ),
                )
            else:
                accessor_string = argparse_prefix + (
                    fieldname if argparse_prefix == "" else "." + fieldname
                )
                has_map_from_namespace = (
                    config_fields_mapped_from_namespace_fields
                    and accessor_string in config_fields_mapped_from_namespace_fields
                )

                choices = get_args(fieldtype) if get_origin(fieldtype) is Literal else None
                # allow null as one of the argparse choices if the field was an Optional
                if choices and found_none_in_union:
                    choices = choices + (None,)
                parent_argparser_or_group.add_argument(
                    "--" + accessor_string,
                    action=_ThronfigArgparseAction,
                    type=_ArgTypeReader(fieldtype, allow_parsing_none_and_null),
                    dest=None,
                    # nargs="*" if is_seq else None, # hmm, should we do this
                    nargs=None,
                    required=this_arg_is_required if not has_map_from_namespace else False,
                    choices=choices,
                )

    def patch_from_command_line_args(
        self,
        args: Optional[Sequence[str]] = None,
        extend_existing_parser: Optional[argparse.ArgumentParser] = None,
        map_namespace_fields_to_config_fields: Optional[Mapping[str, str]] = None,
        allow_parsing_none_and_null=True,
        ignore_unknown_fields=True,
        expect_all_args_to_be_known=True,
        top_level_argparse_prefix: str = "",
        prog=None,
        usage=None,
        description=None,
        epilog=None,
        parents=[],
        formatter_class=argparse.HelpFormatter,
        prefix_chars="-",
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler="error",
        add_help=True,
        allow_abbrev=False,
    ) -> Sequence[str]:
        parser = (
            argparse.ArgumentParser(
                prog,
                usage,
                description,
                epilog,
                parents,
                formatter_class,
                prefix_chars,
                fromfile_prefix_chars,
                argument_default,
                conflict_handler,
                add_help,
                allow_abbrev,
            )
            if extend_existing_parser is None
            else extend_existing_parser
        )
        self.__register_argparsers_recursively(
            parser,
            top_level_argparse_prefix,
            tuple(map_namespace_fields_to_config_fields.values())
            if map_namespace_fields_to_config_fields
            else None,
            allow_parsing_none_and_null=allow_parsing_none_and_null,
        )

        namespace, remains = _parse_args_mine(parser, args, expect_all_args_to_be_known)
        # if map_namespace_fields_to_config_fields is present, we patch those into the
        # namespace._patch_dict now, might cover some no-default config fields
        if map_namespace_fields_to_config_fields:
            for (
                namespace_field,
                config_field,
            ) in map_namespace_fields_to_config_fields.items():
                if not hasattr(namespace, namespace_field):
                    raise ConfigArgumentPatchingError(
                        f"cannot map namespace field {namespace_field} to config field {config_field}: parser namespace has no such field '{namespace_field}'"
                    )
                _ThronfigArgparseAction.read_in_option_strings(
                    namespace, getattr(namespace, namespace_field), "--" + config_field
                )

        self.__replace_shallow(
            replace(
                self.from_dict(
                    self.to_dict(),
                    patch_dict=namespace._patch_dict,
                    ignore_unknown_fields=ignore_unknown_fields,
                )
            )
        )
        return remains

    @classmethod
    def parse_args(
        cls: Type[ThronfigSubclassT],
        config_argname: str,
        args: Optional[Sequence[str]] = None,
        extend_existing_parser: Optional[argparse.ArgumentParser] = None,
        map_namespace_fields_to_config_fields: Optional[Mapping[str, str]] = None,
        allow_parsing_none_and_null=True,
        ignore_unknown_fields=True,
        expect_all_args_to_be_known=True,
        top_level_argparse_prefix: str = "",
        prog=None,
        usage=None,
        description=None,
        epilog=None,
        parents=[],
        formatter_class=argparse.HelpFormatter,
        prefix_chars="-",
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler="error",
        add_help=True,
        allow_abbrev=False,
    ) -> Tuple[ThronfigSubclassT, argparse.Namespace]:
        """
        one-liner to call in your main function. This will
        - build/extend an ArgumentParser with a long argument (e.g. --argname) named
          config_argname that takes a config file path and is to be parsed as the class type
        - adds command line arguments that, when specified, will patch/override values in
          the file-loaded config of the options, i.e. --sub.sub1.field1 1234 will override
          the field "field1" in the sub-config at field "sub1" in the subconfig at field
          "sub" with the value 1234.

        The `map_namespace_fields_to_config_fields` is a mapping of (key, value) where
        namespace.{key} will be filled in as the value for the --{value} optional argument.
        (Note that you shouldn't specify the '--' prefix in the mapping.) This is useful for
        when there are multiple ways in the argparse CLI to satisfy/fill in a config field
        (i.e. a positional argument)

        Note that this function is more 'fused' than merely being a composition of
        cls.load_from_json_file and cls.patch_from_command_line_args. It will apply the
        command line patches directly on the parsed dict from the json file, and so
        typechecking happens only once on the final patched dataclass. Contrast that to
        using cls.load_from_json_file and then cls.patch_from_command_line_args separately,
        which will trigger a typecheck after each step.
        """
        parser = (
            argparse.ArgumentParser(
                prog,
                usage,
                description,
                epilog,
                parents,
                formatter_class,
                prefix_chars,
                fromfile_prefix_chars,
                argument_default,
                conflict_handler,
                add_help,
                allow_abbrev,
            )
            if extend_existing_parser is None
            else extend_existing_parser
        )
        config_file_arg = parser.add_argument(
            f"-{config_argname[0]}",
            f"--{config_argname}",
            type=str,  # this used to be FileType, but that's deprecated in py 3.14
            help="config file",
        )
        parser_without_patch_options = deepcopy(parser)
        cls.__register_argparsers_recursively(
            parser,
            top_level_argparse_prefix,
            allow_parsing_none_and_null=allow_parsing_none_and_null,
            config_fields_mapped_from_namespace_fields=tuple(
                map_namespace_fields_to_config_fields.values()
            )
            if map_namespace_fields_to_config_fields
            else None,
        )

        namespace, remains = _parse_args_mine(parser, args, expect_all_args_to_be_known)

        # if map_namespace_fields_to_config_fields is present, we patch those into the
        # namespace._patch_dict now, might cover some no-default config fields
        def __patch_namespace_with_mapping(_namespace):
            if map_namespace_fields_to_config_fields:
                for (
                    namespace_field,
                    config_field,
                ) in map_namespace_fields_to_config_fields.items():
                    if not hasattr(_namespace, namespace_field):
                        raise ConfigArgumentPatchingError(
                            f"cannot map namespace field {namespace_field} to config field {config_field}: parser namespace has no such field '{namespace_field}'"
                        )
                    _ThronfigArgparseAction.read_in_option_strings(
                        _namespace, getattr(namespace, namespace_field), "--" + config_field
                    )

        __patch_namespace_with_mapping(namespace)

        if (config_file_arg_result := getattr(namespace, config_argname)) is None:
            # this means no config file was loaded.
            loaded_dict = {}
            # however, the whole config might already be present inside the default dict
            # together with namespace._patch_dict, which we could try and convert right away
            # to save some effort; if anything is missing we should get a typecheck err
            try:
                return (
                    cls.from_dict(
                        loaded_dict,
                        patch_dict=namespace._patch_dict,
                        ignore_unknown_fields=ignore_unknown_fields,
                    ),
                    namespace,
                )
            except ConfigDoesNotTypecheck:
                # okay NOW we're sure that we should redo argparse with args mandatory to
                # give a better error message telling the user to supply all args if no file
                pass

            # redo the argparse, except now all the options are mandatory. The whole config
            # will now come from command line args since there was no config file loaded!
            parser = parser_without_patch_options
            cls.__register_argparsers_recursively(
                parser,
                top_level_argparse_prefix,
                tuple(map_namespace_fields_to_config_fields.values())
                if map_namespace_fields_to_config_fields
                else None,
                allow_parsing_none_and_null=allow_parsing_none_and_null,
                args_required_if_no_default=True,
            )

            def __override_parser_error_func(message: str):
                from gettext import gettext

                parser.print_usage(sys.stderr)
                parser.exit(
                    2,
                    gettext(f"{parser.prog}: error: {message}\n")
                    + f"    (All config field --options without a default value declared are now required, due to config filepath argument {'/'.join(config_file_arg.option_strings)} being not specified and the defaults not being sufficient)\n",
                )

            parser.error = __override_parser_error_func
            namespace, remains = _parse_args_mine(parser, args, expect_all_args_to_be_known)
            __patch_namespace_with_mapping(namespace)
        else:
            with _open_fname_or_file(
                config_file_arg_result, "r", treat_hyphen_path_as_stdin=True
            ) as f:
                loaded_dict = json.load(f)

        return (
            cls.from_dict(
                loaded_dict,
                patch_dict=namespace._patch_dict,
                ignore_unknown_fields=ignore_unknown_fields,
            ),
            namespace,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json_string(self, pretty: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=None if not pretty else 2)

    def save_to_json_file(
        self,
        fname_or_file: Union[str, os.PathLike, TextIO],
    ):
        with _open_fname_or_file(fname_or_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# then a final goodie inspired by pyrallis: a main-wrapper that auto-populates some defaults
# and sets up everything incl. patch args and an arg to read config from file
def thronfigure(main_func: Callable[[ThronfigSubclassT], _T]) -> Callable[[], _T]:
    config_argname = None
    config_type: Optional[Type[ThronfigSubclassT]] = None
    already_chosen = False
    for key, ty in main_func.__annotations__.items():
        if key == "return":
            continue
        if not issubclass(ty, Thronfig):
            continue
        # first ty that is a subclass of Thronfig is taken to be the config arg
        if already_chosen:
            # this means there was a second arg that is also annotated Thronfig
            raise TypeError(
                f"Function wrapped by @thronfigure contains more than one input argument annotated with a Thronfig subclass! It must have only one."
            )
        config_argname = key
        config_type = ty  # type: ignore
        already_chosen = True

    if config_argname is None or config_type is None:
        raise TypeError(
            "Function wrapped by @thronfigure must contain exactly one input argument annotated with a Thronfig subclass type"
        )

    @wraps(main_func)
    def wrapped():
        return main_func(config_type.parse_args(config_argname)[0])

    return wrapped


"""
# Example:

from typing import Sequence, Optional
from dataclasses import dataclass
from thronf import Thronfig, thronfigure


@dataclass
class Subfig(Thronfig):
    a: str

@dataclass
class Testfig(Thronfig):
    a: int
    b: Sequence[int]
    c: Optional[Subfig]
    d: Optional[Sequence[Optional[Subfig]]]
    e: Sequence[Sequence[int]]
    f: Sequence[Sequence[Subfig]]

# The fastest way to use thronfig is like so. The name of the Thronfig-subclass-typed arg to
# the decorated function will be used as the option string for specifying a config file to
# load, and we'll also get command line options to override every field in the specified
# dataclass. e.g. with this, we'll get --config (-c), --a, --b, --c.a, --d, --e, --f and
# --help (-h)

@thronfigure
def main(config: Testfig):
    print(config)

if __name__ == "__main__":
    main()

# then run with e.g. (here we cheese by piping the config into stdin rather than from disk)
 
$ echo '{"a": 2, "b": [2,3], "c": null, "d": [{"a": "hi"}], "e": [[2,3],[4,5]], "f": [[{"a": "fo"},{"a":"fi"}],[{"a": "fafa"},{"a": "fifi"}]]}' | python thronf_test.py -c - --a 2 --c.a hello --e [[1,2,5],[3,4,5]]

# This prints:
# Testfig(a=2, b=[2, 3], c=Subfig(a='hello'), d=[Subfig(a='hi')], e=[[1, 2, 5], [3, 4, 5]], f=[[Subfig(a='fo'), Subfig(a='fi')], [Subfig(a='fafa'), Subfig(a='fifi')]])

# Other ways to load a config and patch values onto a config:
#     - <your Thronfig subclass>.parse_args(...)
#     will create (or extend) an ArgumentParser, adding command line options for reading in a file
#     from path or stdin (-), as well as command line options that, when specified, will
#     override values on top of those from the file.

#     - <your Thronfig subclass instance>.patch_from_command_line_args(...) 
#     will read command line args to override values on an existing config instance
"""
