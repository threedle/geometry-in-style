# Threedle Logger: Thlogger
# simple logger + a polyscope session recorder for recording polyscope calls on
# headless environments for later replaying

__all__ = [
    "Thlogger",
    "Loglevel",
    "LOG_NONE",
    "LOG_INFO",
    "LOG_DEBUG",
    "LOG_TRACE",
    "VIZ_NONE",
    "VIZ_INFO",
    "VIZ_DEBUG",
    "VIZ_TRACE",
    "PSRSpecialArray",
    "_PolyscopeRegisteredStructProxy",
]

###############################################################################
# logger code #################################################################
from typing import Callable, Optional, List, TypeVar, Any
import polyscope as ps
import os
import sys

LOG_NONE = 0
LOG_INFO = 1
LOG_DEBUG = 3
LOG_TRACE = 5

VIZ_NONE = 10
VIZ_INFO = 11
VIZ_DEBUG = 13
VIZ_TRACE = 15

# type alias
Loglevel = int
# type var
_T = TypeVar("_T")

THLOG_PREFIXES = {LOG_INFO: "INFO ", LOG_DEBUG: "DEBUG", LOG_TRACE: "TRACE"}

THLOG_STR_TO_LOGLEVEL = {"NONE": 0, "INFO": 1, "DEBUG": 3, "TRACE": 5}


def read_loglevel_str(s: Optional[str], isviz: bool) -> Optional[int]:
    if s is not None:
        if s not in THLOG_STR_TO_LOGLEVEL:
            raise ValueError(f"Unknown thlog loglevel string {s}")
        return THLOG_STR_TO_LOGLEVEL[s] + (VIZ_NONE if isviz else LOG_NONE)
    return s


class Thlogger:
    def __init__(
        self,
        loglevel: Loglevel,
        vizlevel: Loglevel,
        moduleprefix: str,
        imports: List["Thlogger"] = [],
        propagate_to_imports: bool = True,
    ):
        self.loglevel = (
            read_loglevel_str(os.environ.get("THLOG_LOG"), isviz=False) or loglevel
        )
        """
        Allows overriding logging level by setting the env var THLOG_LOG, whose value can be
        NONE, INFO, DEBUG, or TRACE.
        """
        self.allow_ps_init = not bool(os.environ.get("NO_POLYSCOPE"))
        self.vizlevel = (
            read_loglevel_str(os.environ.get("THLOG_VIZ"), isviz=True) or vizlevel
        )
        """
        Allows overriding viz level by setting the env var THLOG_VIZ, whose value can be
        NONE, INFO, DEBUG, or TRACE.
        """
        self.to_stdout_not_stderr = os.environ.get("THLOG_STDOUT") is not None
        """
        Allows overriding the destination file to be stdout rather than stderr (default)
        by setting the env var THLOG_STDOUT.
        """
        self.dest_file = None if self.to_stdout_not_stderr else sys.stderr
        self.moduleprefix = moduleprefix
        """
        Prefix for this module's thlog printout messages
        """
        self.imported_thloggers = imports
        """
        Imported Thlogger objects from other modules, to broadcast log levels and unify
        all modules to use the same Polyscope recorder
        """

        # this is only True if ps.init() was called. if we are running in
        # PolyscopeRecorder-only mode, this remains False
        self.ps_initialized = False
        """
        Whether ps.init() was actually called. PolyscopeRecorder still functions regardless.
        """

        if propagate_to_imports:
            for thlogger in self.imported_thloggers:
                thlogger.loglevel = self.loglevel
                thlogger.vizlevel = self.vizlevel
                thlogger.allow_ps_init = self.allow_ps_init
        else:
            # in case the bool is not set but the list still contains imports
            self.imported_thloggers = []

        # this will hold a PolyscopeRecorder when init_polyscope(start_polyscope_recorder=True)
        self.__psr = None
        # this will hold a PolyscopePlotter
        self.__psplt = None

    def set_levels(
        self, loglevel: Loglevel, vizlevel: Loglevel, propagate_to_imports: bool = False
    ):
        """
        Set this Thlogger's logging and visualization levels. If
        propagate_to_imports is True, then all Thloggers imported by this one
        will recursively run set_levels, setting levels on themselves and their
        own imported Thloggers.
        """
        self.loglevel = loglevel
        self.vizlevel = vizlevel
        if propagate_to_imports:
            for thlogger in self.imported_thloggers:
                thlogger: Thlogger
                thlogger.set_levels(
                    loglevel, vizlevel, propagate_to_imports=propagate_to_imports
                )

    def __set_ps_initialized(self, ps_initialized: bool, broadcast: bool = True):
        self.ps_initialized = ps_initialized
        if self.__psr:
            self.__psr.ps_initialized = ps_initialized
        if broadcast:
            for thlogger in self.imported_thloggers:
                thlogger.__set_ps_initialized(ps_initialized, broadcast=broadcast)

    @property
    def psr(self):
        """
        a frontend for the polyscope module that records every call, which can be
        saved out at the end with thlog.save_ps_recording(filename).
        """
        if self.__psr:
            return self.__psr
        else:
            return ps

    @property
    def psplt(self):
        """
        a frontend for a polyscope plotter (which works with polyscope recorder too if enabled)
        """
        if self.__psplt:
            return self.__psplt
        else:
            raise PolyscopePlotterNotInitialized("call thlog.init_polyscope_plotter first")

    def save_ps_recording(self, ps_recording_npz_fname: str, comment: str = ""):
        """
        If the polyscope recorder was started, then this will save the recording to a file.
        If it was not, then this will do nothing (with a message saying so)

        `comment` is a string to save along with the replay file. You might use this
        string to store config info or metadata about the run that produced the recording.
        """
        if self.__psr:
            self.info(f"saving the polyscope recording to {ps_recording_npz_fname}")
            self.__psr._save_ps_recording(ps_recording_npz_fname, comment=comment)
        else:
            self.info(
                f"polyscope recorder was not started, NO recording will be saved to {ps_recording_npz_fname}"
            )

    def play_ps_recording(
        self,
        ps_recording_npz_fname: str,
        screenshot_filename_template: Optional[str] = None,
        draw_playback_ui: bool = True,
    ):
        """
        plays back a polyscope recording npz file
        """
        if not self.allow_ps_init:
            raise EnvironmentError(
                f"Cannot play back a polyscope recording on an environment where initializing polyscope with ps.init() has been blocked by the NO_POLYSCOPE environment variable. Did you set the environment variable NO_POLYSCOPE by mistake? Its value is currently {repr(os.getenv('NO_POLYSCOPE'))}. Must unset or set as empty string."
            )
        return PolyscopeRecorder._replay(
            self,
            ps_recording_npz_fname,
            screenshot_filename_template=screenshot_filename_template,
            draw_playback_ui=draw_playback_ui,
        )

    def log_in_ps_recording(self, message: str, print_live_too: bool = False):
        """
        save a message to be printed out when the polyscope recording playback
        reaches this point
        if print_live_too,  will also print the message during the live code run
        that is generating the polyscope recording
        """
        if self.__psr:
            self.__psr._store_call(_PSREC_PRINT_TAG, (message,), {})
            if print_live_too:
                self.info(f"(logged to polyscope recording):\n{message}")

    def __set_ps_recorder(self, psr: "PolyscopeRecorder", broadcast: bool = True):
        """
        note that this will reset the polyscope recorder if it is already init'd
        """
        self.__psr = psr
        if broadcast:
            for thlogger in self.imported_thloggers:
                thlogger.__set_ps_recorder(psr, broadcast=broadcast)

    def __set_ps_plotter(self, psplt: "PolyscopePlotter_v1", broadcast: bool = True):
        self.__psplt = psplt
        if broadcast:
            for thlogger in self.imported_thloggers:
                thlogger.__set_ps_plotter(psplt, broadcast=broadcast)

    def init_polyscope_plotter(self, propagate_to_imports: bool = True, version: int = 1):
        """
        best used after init_polyscope(...,start_polyscope_recorder=True)
        but can still work without the recorder active
        """
        if version == 1:
            self.__set_ps_plotter(PolyscopePlotter_v1(self), broadcast=propagate_to_imports)
        else:
            raise ValueError(f"unknown polyscope plotter version {version}")

    def init_polyscope(
        self, propagate_to_imports: bool = True, start_polyscope_recorder: bool = False
    ):
        """
        "Initialize" polyscope and/or the polyscope recorder.

        What this actually does depends on a few things.
        - If the environment variable NO_POLYSCOPE was set to anything nonempty
          before this Thlogger object was initialized, then ps.init() will *not*
          be run. This is helpful for use on a compute cluster or other headless
          environment where ps.init (which expects a display for GLFW) will
          crash.
        - If start_polyscope_recorder is True, then regardless of the above, a
          new polyscope recording session will be initialized. The polyscope
          recorder facilities do not rely on ps.init() having been run, so
          NO_POLYSCOPE set to any nonempty value and start_polyscope_recorder
          set to True is the intended usage for headless environments.
        - If both NO_POLYSCOPE is nonempty (causing ps.init to not run) and
          start_polyscope_recorder is False (so that there is no recorder
          session), then there is nothing to do.

        propagate_to_imports=True will do all this recursively to all Thlogger
        objects imported from other modules imported by your code (so that there
        is only one global polyscope recorder session, recording all polyscope
        calls across all your modules.)
        """
        if start_polyscope_recorder:
            self.__set_ps_recorder(PolyscopeRecorder(), broadcast=propagate_to_imports)

        # if environment says not to ps init, and there is no polyscope recorder, then there
        # is nothing to do
        if not self.allow_ps_init and not self.__psr:
            return

        # else, if not already initialized and we're allowed to init, do it
        if not self.ps_initialized and self.allow_ps_init:
            ps.init()
            self.__set_ps_initialized(True, broadcast=propagate_to_imports)

    def log(self, loglevel: Loglevel, message: str, flush: bool = False):
        """
        Logs a message based on the current text logging level. If the message's
        requested logging level is more verbose the current logging level (i.e.
        if log is called with LOG_TRACE but the script's Thlogger is running at
        LOG_INFO), the message will not show up.
        """
        if self.loglevel >= loglevel:
            if loglevel not in THLOG_PREFIXES:
                prefix = "  !  "
            else:
                prefix = THLOG_PREFIXES[loglevel]
            print(
                f"[{self.moduleprefix} | {prefix}] {message}",
                flush=flush,
                file=self.dest_file,
            )

    def info(self, msg: str, flush: bool = False):
        """
        Logs a message at the LOG_INFO level.
        """
        return self.log(LOG_INFO, msg, flush=flush)

    def debug(self, msg: str, flush: bool = False):
        """
        Logs a message at the LOG_DEBUG level.
        """
        return self.log(LOG_DEBUG, msg, flush=flush)

    def trace(self, msg: str, flush: bool = False):
        """
        Logs a message at the LOG_TRACE level. (This is the most verbose level)
        """
        return self.log(LOG_TRACE, msg, flush=flush)

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    def do(
        self, vizlevel: Loglevel, fn: Callable[[], _T], needs_polyscope: bool = False
    ) -> Optional[_T]:
        """
        Run a callback function at a given visualization level. (If the
        requested vizlevel is VIZ_DEBUG, and the current Thlogger's viz level is
        VIZ_INFO, the callback will not run.) If the callback is run, returns
        its result.
        """
        if self.guard(vizlevel, needs_polyscope=needs_polyscope):
            return fn()
        return None

    def err(self, message: str):
        """
        Logs an error message. This bypasses all loglevel (verbosity level) checks.
        """
        print(f"[{self.moduleprefix} | ERROR] {message}", flush=True)

    def logguard(self, loglevel: Loglevel) -> bool:
        """
        Returns True if this Thlogger's text-logging level is at least as
        verbose as the requested text logging level
        """
        return self.loglevel >= loglevel

    def guard(
        self,
        vizlevel: Loglevel,
        needs_polyscope: bool = False,
        allow_polyscope_recorder: bool = True,
    ) -> bool:
        """
        Returns True if this Thlogger's visualization level is at least as
        'verbose' as the requested visualization level. You can indicate
        needs_polyscope=True if the code guarded by this conditional contains
        polyscope calls; if `allow_polyscope_recorder == True`, just having the
        polyscope recorder active will let this evaluate to True and let that
        code run too (to be recorded into a polyscope recording), even if
        `ps.init()` was never called.
        """
        return bool(
            (self.vizlevel >= vizlevel)
            and (
                (not needs_polyscope)
                or (needs_polyscope and self.ps_initialized)
                or (needs_polyscope and allow_polyscope_recorder and self.__psr)
            )
        )


###############################################################################
# plotting utilities using matplotlib in polyscope (matplotlib only imported if both
# ps.init() is run and plotter is initialized; this means it's indeed possible to make
# recordings with plot calls on a system without matplotlib if ps.init() is never
# initialized, but the polyscope recorder is initialized)
from typing import Sequence, Dict, Union
import numpy as np
import io


class PolyscopePlotterNotInitialized(BaseException):
    pass


class PolyscopePlotter_v1:
    def __init__(self, thlog: Thlogger):
        self.arrays: Dict[str, np.ndarray] = {}
        self.__thlog = thlog
        self.__plt = None  # will hold matplotlib.pyplot

    def _store_call(
        self, function_name: str, fn_args: Sequence[Any], fn_kwargs: Dict[str, Any]
    ):
        if isinstance((psr := self.__thlog.psr), PolyscopeRecorder):
            psr._store_call(
                function_name, fn_args, fn_kwargs, as_psplt_method_of_psplt_version=1
            )

    def init_scalar_quantity(self, quantity_name: str, array: np.ndarray):
        self._store_call("init_scalar_quantity", (quantity_name, array), {})
        if array.ndim != 1:
            raise ValueError("array must be 1D for init_scalar_quantity")
        self.arrays[quantity_name] = array.copy()
        # copy so that self.modify_scalar_quantity doesn't edit the psrec-saved array above
        # (we store all inits and modifies of that array, not just the last modified version)

    def modify_scalar_quantity(
        self, quantity_name: str, index: int, value: Union[int, float]
    ):
        self._store_call("modify_scalar_quantity", (quantity_name, index, value), {})
        self.arrays[quantity_name][index] = value

    def extend_scalar_quantity(self, quantity_name: str, array_to_concat: np.ndarray):
        self._store_call("extend_scalar_quantity", (quantity_name, array_to_concat), {})
        old_arr = self.arrays[quantity_name]
        # all our arrays are 1D so this is fine
        self.arrays[quantity_name] = np.concatenate((old_arr, array_to_concat), axis=0)

    def add_ps_plot(
        self,
        plot_name: str,
        quantity_names: Sequence[str],
        x_axis_quantity_name: Optional[str] = None,
        dpi: float = 80,
    ):
        self._store_call(
            "add_ps_plot",
            (plot_name, quantity_names),
            {"x_axis_quantity_name": x_axis_quantity_name, "dpi": dpi},
        )
        # only run the rest if polyscope is initialized; see below for reason...
        if not self.__thlog.ps_initialized:
            return
        # okay we're actually doing this
        if self.__plt is None:
            import matplotlib.pyplot as plt

            self.__plt = plt
        # if the x axis quantity is None, use an arange
        quantities = tuple(self.arrays[name] for name in quantity_names)
        max_length = max(map(lambda arr: len(arr), quantities))
        if x_axis_quantity_name is None:
            x_axis = np.arange(max_length)
        else:
            x_axis = self.arrays[x_axis_quantity_name]
        # make a matplotlib figure
        self.__thlog.trace("making matplotlib figure")
        fig, ax = self.__plt.subplots(dpi=dpi)
        fig.tight_layout()
        for quantity_name, quantity in zip(quantity_names, quantities):
            ax.plot(x_axis, quantity, label=quantity_name)
            ax.set_ylabel(quantity_name)
            if x_axis_quantity_name:
                ax.set_xlabel(x_axis_quantity_name)
            ax.legend()
        # this is why this method proceeds past storing itself only when ps.init() is run.
        # we don't record this, (the image may be very large and would be bad to save it in
        # all psrec) so we record this add_ps_plot method call and generate the plot and add
        # the image LIVE during live polyscope-enabled runs or during recording playback
        ps.add_color_image_quantity(
            plot_name, PolyscopePlotter_v1.fig_to_image_array(fig)[:, :, :3], enabled=True
        )

    @staticmethod
    def fig_to_image_array(figure) -> np.ndarray:
        with io.BytesIO() as buf:
            figure.savefig(buf, format="raw", dpi=figure.get_dpi())
            buf.seek(0)
            arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image = arr.reshape((h, w, -1)).astype(np.float32) / 255.0
        return image


###############################################################################
# polyscope recorder code #####################################################
from typing import Dict, Union, Tuple, Sequence
from collections import namedtuple
import numpy as np
import json
import io
import polyscope.imgui as psim


class _PolyscopeRegisteredStructProxyClassMethod:
    def __init__(
        self,
        method_name: str,
        belonging_to_struct_proxy: "_PolyscopeRegisteredStructProxy",
        method_fn: Optional[Callable[..., Any]] = None,
    ):
        self.method_name = method_name
        self.belonging_to_struct_proxy = belonging_to_struct_proxy
        self.method_fn = method_fn
        # method_fn should be not-None when belonging_to_struct_proxy has a non-None
        # real_ps_struct (which is when ps is initialized in the environment)

    def __call__(self, *args, **kwargs):
        args_normalized, kwargs_normalized = (
            self.belonging_to_struct_proxy.registered_with_hooked_fn.recorder._store_call(
                self.method_name,
                args,
                kwargs,
                as_class_method_of_registered_struct_proxy=self.belonging_to_struct_proxy,
            )
        )
        if self.method_fn is not None:
            return self.method_fn(*args_normalized, **kwargs_normalized)


class _PolyscopeRegisteredStructProxy:
    def __init__(
        self, registered_with_hooked_fn: "_PolyscopeHookedFunction", id_in_recorder: int
    ):
        self.real_ps_struct: Optional[ps.Structure] = None
        self.registered_with_hooked_fn = registered_with_hooked_fn
        self.id_in_recorder = id_in_recorder

    def __getattr__(self, name, /):
        if not self.real_ps_struct:
            # we cannot tell if this is callable or not, since when ps is not initialized we
            # don't even have the structure to check and get the function! so we'll have to
            # assume that it is callable, and always return a proxy class method
            return _PolyscopeRegisteredStructProxyClassMethod(name, self)
        else:
            ps_struct_attr = getattr(self.real_ps_struct, name)
            if callable(ps_struct_attr):
                return _PolyscopeRegisteredStructProxyClassMethod(
                    name, self, ps_struct_attr
                )
            else:
                return ps_struct_attr


class _PolyscopeHookedFunction:
    def __init__(
        self, fn: Callable[..., Any], function_name: str, recorder: "PolyscopeRecorder"
    ):
        self.fn = fn
        self.function_name = function_name
        self.recorder = recorder

        if function_name.startswith("register_"):
            # EXTREME HACK BODGE PLUS: currently, all ps.register_* methods return a
            # Structure that is usually intended to have its methods called as well. We want
            # a way to record the structure that is returned, and make the same calls at
            # replay.
            self.registered_struct_proxy = self.recorder._make_registered_struct_proxy(self)
        else:
            self.registered_struct_proxy = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args_normalized, kwargs_normalized = self.recorder._store_call(
            self.function_name, args, kwargs
        )

        if self.recorder.ps_initialized:
            real_ps_result = self.fn(*args_normalized, **kwargs_normalized)
            if self.registered_struct_proxy:
                # this ps function is a register_* function, so we return our proxy.
                # moreover, in this situation, ps is initialized so we can give the proxy
                # the real struct returned by polyscope for the proxy classmethod to
                # actually call the real classmethod
                self.registered_struct_proxy.real_ps_struct = real_ps_result
                return self.registered_struct_proxy
            else:
                # not a register_* function, no need for struct proxy
                return real_ps_result
        else:
            if self.registered_struct_proxy:
                # is a register_* function, but not in a polyscope-initialized environment,
                # so just return the bare proxy
                return self.registered_struct_proxy
            else:
                return None


class PolyscopeRecordingPlaybackError(BaseException):
    pass


_PSREC_ARRAY_TAG = "PSREC_ARRAY"
_PSREC_PROXY_TAG = "PSREC_STRUCT_PROXY"
_PSREC_PRINT_TAG = "PSREC_PRINT"
_PSREC_PSPLT_TAG = "PSREC_PSPLT"
# we also record calls to our own polyscope matplotlib plotter


class PSRSpecialArray:
    """
    wraps an np.ndarray with special psrec-specific metadata indicating special processing
    such as converting between the saved array and the array given to polyscope calls.

    the use case is for registering image quantities in polyscope. We'd like to
    save a compressed png in the recording (as an array containing the png file bytes)
    but transparently pass the original image array to any passthru polyscope calls,
    and do the conversion to pass along to polyscope at playback time.
    """

    def __init__(
        self, array_for_ps: np.ndarray, array_for_saving: np.ndarray, metadata: dict
    ):
        # metadata has to be serializable!
        self.array_for_ps = array_for_ps
        self.array_for_saving = array_for_saving
        self.metadata = metadata

    @classmethod
    def image_array_as_png_bytes(cls, image: np.ndarray):
        """
        converts the (h, w, channels) image into a stream of png bytes; this byte array is
        what gets saved as the image array in any polyscope recorded function call that uses
        it, in order to save space in the ps recording. It is transparently converted to the
        raw image array at playback time / runtime when a polyscope call is actually run.

        assumes rgb (h, w, 3) or rgba (h, w, 4), float values in range [0,1]

        requires PIL!
        """
        assert image.ndim == 3
        assert image.shape[2] in (3, 4)
        assert np.issubdtype(image.dtype, np.floating), (
            "array must be a numpy floating point array"
        )
        from PIL import Image

        mode = "RGB" if image.shape[2] == 3 else "RGBA"
        image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        png_bytes_buf = io.BytesIO()
        Image.fromarray(image_uint8).save(png_bytes_buf, format="PNG")
        png_bytes_arr = np.array(png_bytes_buf.getvalue())  # np.array on a bytes value
        return cls(
            array_for_ps=image, array_for_saving=png_bytes_arr, metadata={"pngbytes": mode}
        )

    @staticmethod
    def imgformat_bytes_to_array(imgformat_bytes_arr: np.ndarray) -> np.ndarray:
        from PIL import Image

        a = np.asarray(Image.open(io.BytesIO(imgformat_bytes_arr.tobytes())))
        return a.astype(np.float32) / 255.0


def _resolve_psrec_tagged_array(
    __arrays: Dict[str, np.ndarray], idx: int, metadata: Optional[dict]
) -> np.ndarray:
    ret_array = __arrays[f"{_PSREC_ARRAY_TAG}_{idx}"]
    if metadata:
        if pil_mode := metadata.get("pngbytes"):
            ret_array = PSRSpecialArray.imgformat_bytes_to_array(ret_array)
        # other array processing according to metadata here
    return ret_array


def _make_tagged_array_key(idx: int):
    return f"{_PSREC_ARRAY_TAG}_{idx}"


class PolyscopeRecorder:
    """
    WARNING: EXTREME BODGE, no guarantees that this will work forever and across all ps
    module-level functions!!

    a class which allows access to module-level functions from the polyscope module but
    intercepts them to record inputs to the calls to a file before running the function.
    There are some extra custom affordances for the register_* functions, returning a
    "reference" that can have the original structure methods called on them.

    The PolyscopeRecorder is useful in cases where polyscope is NOT initialized (due to
    being run on a headless environment, etc) or we just want a "replay file" that records
    all calls to polyscope to replay and review.
    """

    def __init__(self):
        # store the arrays used in the intercepted ps calls
        self.arrays: Dict[int, np.ndarray] = {}
        self.metadata_dicts: Dict[int, dict] = {}
        self.curr_array_id = 0
        self.curr_registered_struct_proxy_id = 0
        self.calls: List[str] = []

        # if True, log calls + also run them with real polyscope
        self.ps_initialized = False

    def _make_registered_struct_proxy(
        self, via_hooked_fn: _PolyscopeHookedFunction
    ) -> _PolyscopeRegisteredStructProxy:
        registered_struct_proxy_id = self.curr_registered_struct_proxy_id
        registered_struct_proxy = _PolyscopeRegisteredStructProxy(
            via_hooked_fn, registered_struct_proxy_id
        )
        registered_struct_proxy.id_in_recorder = registered_struct_proxy_id
        self.curr_registered_struct_proxy_id += 1
        return registered_struct_proxy

    def _store_array(
        self, array_or_arraywithmeta: Union[np.ndarray, PSRSpecialArray]
    ) -> int:
        array_id = self.curr_array_id
        if isinstance(array_or_arraywithmeta, PSRSpecialArray):
            self.arrays[array_id] = array_or_arraywithmeta.array_for_saving
            self.metadata_dicts[array_id] = array_or_arraywithmeta.metadata
        else:
            self.arrays[array_id] = array_or_arraywithmeta
        self.curr_array_id += 1
        return array_id

    def _store_call(
        self,
        function_name: str,
        fn_args: Sequence[Any],
        fn_kwargs: Dict[str, Any],
        as_class_method_of_registered_struct_proxy: Optional[
            _PolyscopeRegisteredStructProxy
        ] = None,
        as_psplt_method_of_psplt_version: Optional[int] = None,
    ):
        args_serialized: Tuple[Union[Tuple[str, int], Any], ...] = tuple(
            (
                (_PSREC_ARRAY_TAG, self._store_array(arg))
                if isinstance(arg, (np.ndarray, PSRSpecialArray))
                else arg
            )
            for arg in fn_args
        )

        # throws if not serializable. TODO maybe catch and skip recording this call?
        args_serialized_str = json.dumps(args_serialized)

        # return this for polyscope use, PSRECSpecialArray wraps np arrays with
        # metadata for our own purposes, not for polyscope, we extract the normal array
        args_normalized = tuple(
            arg.array_for_ps if isinstance(arg, PSRSpecialArray) else arg for arg in fn_args
        )

        kwargs_serialized: Tuple[Tuple[str, Union[Tuple[str, int], Any]], ...] = tuple(
            (
                kwarg_key,
                (
                    (_PSREC_ARRAY_TAG, self._store_array(kwarg))
                    if isinstance(kwarg, (np.ndarray, PSRSpecialArray))
                    else kwarg
                ),
            )
            for kwarg_key, kwarg in fn_kwargs.items()
        )

        # throws if not serializable. TODO maybe catch and skip recording this call?
        kwargs_serialized_str = json.dumps(kwargs_serialized)

        kwargs_normalized = {
            kwarg_key: (kwarg.array_for_ps if isinstance(kwarg, PSRSpecialArray) else kwarg)
            for kwarg_key, kwarg in fn_kwargs.items()
        }

        if as_class_method_of_registered_struct_proxy is not None:
            # the string tag, a !, then proxy id, then a period, then the function name
            function_name = f"{_PSREC_PROXY_TAG}!{as_class_method_of_registered_struct_proxy.id_in_recorder}.{function_name}"

        elif as_psplt_method_of_psplt_version:
            function_name = (
                f"{_PSREC_PSPLT_TAG}v{as_psplt_method_of_psplt_version}!{function_name}"
            )

        serialized_str = f"{function_name}\n{args_serialized_str}\n{kwargs_serialized_str}"
        self.calls.append(serialized_str)

        # returns valid args kwargs for polyscope to run, (replacing any
        # PSRECSpecialArray with the actual array)
        return args_normalized, kwargs_normalized

    def __getattr__(self, name: str):
        ps_attr = getattr(ps, name)  # this will throw if name not part of ps, which is fine
        if callable(ps_attr):
            # make a hooked function that will get called instead, wrapping the original
            # polyscope module function
            return _PolyscopeHookedFunction(ps_attr, name, self)
        else:
            return ps_attr

    def _save_ps_recording(self, ps_recording_npz_fname: str, comment: str):
        """
        this should be called from thlog.save_ps_recording, and not
        thlog.psr._save_ps_recording (in the case that thlog.psr is a passthru
        to the polyscope module itself, and not this PolyscopeRecorder)
        """
        calls_concatenated = "\r".join(self.calls)
        # for now all entries in metadata_dicts are for arrays, the index corresponds to an
        # array index but we save the keys with a PSREC_ARRAY_ prefix so the metadata_dicts
        # dict should support metadata for more than just arrays if we need to later on...
        metadata_dicts_to_save = {
            _make_tagged_array_key(k): metadata
            for k, metadata in self.metadata_dicts.items()
        }
        metadata_dicts_serialized = (
            json.dumps(metadata_dicts_to_save) if metadata_dicts_to_save else None
        )
        np.savez_compressed(
            ps_recording_npz_fname,
            version=np.array("1"),
            calls=np.array(calls_concatenated),
            comment=np.array(comment),
            **(
                {"metadata_dicts": np.array(metadata_dicts_serialized)}
                if metadata_dicts_serialized
                else {}
            ),
            **{_make_tagged_array_key(k): arr for k, arr in self.arrays.items()},
        )

    def _print_ps_recorder_status(self):
        print(
            f"""
        arrays:
            {self.arrays}
        current array id
            {self.curr_array_id}
        current registered struct proxy id
            {self.curr_registered_struct_proxy_id}
        calls
            {self.calls}
        """
        )

    @staticmethod
    def _replay(
        thlog: Thlogger,
        ps_recording_npz_fname: str,
        screenshot_filename_template: Optional[str] = None,
        draw_playback_ui: bool = True,
    ):
        __log: Callable[[str, bool], None] = lambda s, verbose: thlog.log(
            LOG_TRACE if verbose else LOG_INFO, s
        )
        is_verbose = thlog.logguard(LOG_TRACE)
        __log("Initializing polyscope for the replay", False)
        thlog.init_polyscope(start_polyscope_recorder=False)

        # no need to start a new recorder, we have to use real polyscope regardless
        # to actually do any playback
        # (although maybe starting a new recorder here could be a way to "make a
        # copy" of a recording and embed it in a larger recording?)

        arrays = {}
        registered_structs = []
        __log(
            f"Loading {ps_recording_npz_fname} as polyscope recording file for playback",
            False,
        )
        with np.load(ps_recording_npz_fname) as npz:
            recording_npz_format_version = str(npz["version"])
            if recording_npz_comment := str(npz.get("comment", "")):
                __log(f"Comment in recording:\n{recording_npz_comment}", False)

            # TODO: code to handle versioning. no versions yet, but it is in the npz format
            calls_concatenated = str(npz["calls"])
            metadata_dicts__arr = npz.get("metadata_dicts")
            metadata_dicts: Optional[Dict[str, dict]] = (
                json.loads(str(metadata_dicts__arr))
                if metadata_dicts__arr is not None
                else None
            )
            if not calls_concatenated:
                raise PolyscopeRecordingPlaybackError("recording is empty")
            for key, arr in npz.items():
                if key != "calls":
                    arrays[key] = arr

        def __maybe_get_psrec_tagged_array_idx_and_metadata(a):
            is_psrec_tagged_array = (
                isinstance(a, Sequence)
                and not isinstance(a, str)
                and len(a) == 2
                and a[0] == _PSREC_ARRAY_TAG
                and isinstance(a[1], int)
            )
            # a tagged array is indicated with a tuple of length 2 (_PSREC_ARRAY_TAG, idx)
            if is_psrec_tagged_array:
                idx = int(a[1])
                metadata = (
                    metadata_dicts.get(_make_tagged_array_key(idx))
                    if metadata_dicts
                    else None
                )
                # metadata should be a parsed json dict
                metadata = None if not isinstance(metadata, dict) else metadata
                return idx, metadata
            else:
                return None

        def __iterator_through_showframes():
            n_screenshots = 0
            psplt_initialized_at_version = None
            for call_serialized in calls_concatenated.split("\r"):
                lines = call_serialized.split("\n")
                if not ((n_lines := len(lines)) <= 3 and n_lines > 0):
                    raise PolyscopeRecordingPlaybackError(
                        f"error parsing serialized function call with lines {lines}"
                    )
                # first "line" contains the function name. it is either just a string,
                # or it can be of the form f"{_PSREC_PROXY_TAG}!{registered_struct_proxy_id}.{function_name}"
                # or f"{_PSREC_PSPLT_TAG}v{pspltversion}!{function_name}"
                function_name = lines[0]
                registered_struct_proxy_id = None
                method_is_for_psplt_version = None

                if len((function_name_splits := function_name.split("!"))) == 2:
                    if (tag := function_name_splits[0]) == _PSREC_PROXY_TAG:
                        # the part after the ! is "{registered_struct_proxy_id}.{function_name}"
                        after_bang_splits = function_name_splits[1].split(".")
                        registered_struct_proxy_id = int(after_bang_splits[0])
                        function_name = after_bang_splits[1]
                    elif tag.startswith(_PSREC_PSPLT_TAG):
                        method_is_for_psplt_version = int(tag.split("v")[1])
                        function_name = function_name_splits[1]
                    else:
                        raise PolyscopeRecordingPlaybackError(
                            f"Unknown serialized call tag {tag}"
                        )

                # second line contains args, third contains kwargs, both as json lists. if any
                # arg is actually a numpy array, it will be represented as a json 2-array
                # ["PSREC_ARRAY", <id of the array in the arrays dict>], which will be read as
                # a list/tuple where the first item is our special string tag "PSREC_ARRAY" and
                # the 2nd is a string that is a UUID string.

                # other arg types should already be correct from json parsing, since we restrict
                # (we hope) that polyscope API functions only use, other than np arrays, json
                # serializable and readable values as function args and kwargs.
                args = json.loads(lines[1])
                assert isinstance(args, Sequence) and not isinstance(args, str)
                kwargs = json.loads(lines[2])
                assert isinstance(kwargs, Sequence) and not isinstance(kwargs, str)

                # convert args that are ("PSREC_ARRAY", id str) into the appropriate referenced
                # np array and also obtain prettyprinting strings to print the reconstructed
                # function call in the printout
                args_maybe_psrec_tagged_array_idx_and_meta = tuple(
                    map(__maybe_get_psrec_tagged_array_idx_and_metadata, args)
                )
                kwargs_maybe_psrec_tagged_array_meta = tuple(
                    __maybe_get_psrec_tagged_array_idx_and_metadata(kwarg)
                    for _, kwarg in kwargs
                )
                if is_verbose:
                    args_prettyprint = tuple(
                        (
                            f"<saved array {maybe_arr_idx_and_meta[0]}{(' (meta: ' + str(maybe_arr_idx_and_meta[1]) + ')') if maybe_arr_idx_and_meta[1] else ''}>"
                            if maybe_arr_idx_and_meta
                            else repr(arg)
                        )
                        for arg, maybe_arr_idx_and_meta in zip(
                            args, args_maybe_psrec_tagged_array_idx_and_meta
                        )
                    )
                    args_prettyprint_str = ", ".join(args_prettyprint)
                else:
                    args_prettyprint_str = ""

                args = tuple(
                    (
                        _resolve_psrec_tagged_array(arrays, *maybe_arr_idx_and_meta)
                        if maybe_arr_idx_and_meta
                        else arg
                    )
                    for arg, maybe_arr_idx_and_meta in zip(
                        args, args_maybe_psrec_tagged_array_idx_and_meta
                    )
                )

                if is_verbose:
                    kwargs_prettyprint = tuple(
                        (
                            kwarg_key,
                            (
                                f"<saved array {maybe_arr_idx_and_meta[0]}{(' (meta: ' + str(maybe_arr_idx_and_meta[1]) + ')') if maybe_arr_idx_and_meta[1] else ''}>"
                                if maybe_arr_idx_and_meta
                                else repr(kwarg)
                            ),
                        )
                        for (kwarg_key, kwarg), maybe_arr_idx_and_meta in zip(
                            kwargs, kwargs_maybe_psrec_tagged_array_meta
                        )
                    )
                    kwargs_prettyprint_str = (
                        ", " if kwargs_prettyprint else ""
                    ) + ", ".join(
                        f"{kwarg_key}={kwarg}" for kwarg_key, kwarg in kwargs_prettyprint
                    )
                else:
                    kwargs_prettyprint_str = ""

                kwargs = {
                    kwarg_key: (
                        _resolve_psrec_tagged_array(arrays, *maybe_arr_idx_and_meta)
                        if maybe_arr_idx_and_meta
                        else kwarg
                    )
                    for (kwarg_key, kwarg), maybe_arr_idx_and_meta in zip(
                        kwargs, kwargs_maybe_psrec_tagged_array_meta
                    )
                }
                # if function name is PSREC_PRINT, we do the printing rather than ps
                # (this overrides all other scenarios, incl. function being a module-level function)
                # hopefully polyscope never gets a function named exactly PSREC_PRINT
                if function_name == _PSREC_PRINT_TAG:
                    print(*args, flush=True, file=sys.stderr)
                elif method_is_for_psplt_version is not None:
                    # is a PSPLT method, init the right plotter version if not already
                    if psplt_initialized_at_version is None:
                        thlog.init_polyscope_plotter(version=method_is_for_psplt_version)
                        psplt_initialized_at_version = method_is_for_psplt_version
                    else:
                        # check if the initialized plotter matches this method's version
                        if method_is_for_psplt_version != psplt_initialized_at_version:
                            raise PolyscopeRecordingPlaybackError(
                                f"calls to multiple PolyscopePlotter versions found in recording; initialized version {psplt_initialized_at_version} but found call meant for version {method_is_for_psplt_version}"
                            )
                    # tagged plotter method, call our polyscope plotter
                    psplt_fn = getattr(thlog.psplt, function_name)
                    __log(
                        f"thlog.psplt.{function_name}({args_prettyprint_str}{kwargs_prettyprint_str})",
                        True,
                    )
                    psplt_fn(*args, **kwargs)
                elif registered_struct_proxy_id is None:
                    # then if there is no registered struct proxy, the function
                    # should be a module-level function in polyscope.
                    # pretty print the decoded function call:
                    if fn_is_register_something := function_name.startswith("register_"):
                        __log(
                            f"struct{len(registered_structs)} = ps.{function_name}({args_prettyprint_str}{kwargs_prettyprint_str})",
                            True,
                        )
                    else:
                        __log(
                            f"ps.{function_name}({args_prettyprint_str}{kwargs_prettyprint_str})",
                            True,
                        )

                    # then run the function
                    if function_name == "show":
                        if draw_playback_ui:
                            real_ps_result = None
                            yield
                        else:
                            real_ps_result = ps.show()
                            yield
                    else:
                        ps_fn = getattr(ps, function_name)
                        real_ps_result = ps_fn(*args, **kwargs)

                    if fn_is_register_something:
                        registered_structs.append(real_ps_result)
                    if screenshot_filename_template and function_name == "show":
                        # if ps.show(), and screenshot filename template specified, also save a pic
                        screenshot_fname = screenshot_filename_template.format(
                            f"{n_screenshots:06}"
                        )
                        ps.screenshot(screenshot_fname, transparent_bg=False)
                        __log(f"Took screenshot {screenshot_fname}", False)
                        n_screenshots += 1
                else:
                    # resolve the registered struct proxy id as the actual struct that has been
                    # created at that order in the execution chain by a register_* function
                    struct = registered_structs[registered_struct_proxy_id]
                    # prettyprint the reconstructed call
                    __log(
                        f"struct{registered_struct_proxy_id}.{function_name}({args_prettyprint_str}{kwargs_prettyprint_str})",
                        True,
                    )
                    struct_fn = getattr(struct, function_name)
                    struct_fn(*args, **kwargs)

        stepper = __iterator_through_showframes()

        nonloc_lastframereached = False
        fname_basename = os.path.basename(ps_recording_npz_fname)

        def __ps_callback():
            nonlocal nonloc_lastframereached
            psim.Begin(  # type: ignore
                "Playback controls",
                open=None,
                flags=psim.ImGuiWindowFlags_NoResize  # type: ignore
                | psim.ImGuiWindowFlags_AlwaysAutoResize,  # type: ignore
            )
            psim.TextUnformatted(f"Recording: {fname_basename}")  # type: ignore
            if psim.Button("Step to next show() frame"):  # type: ignore
                try:
                    next(stepper)
                except StopIteration:
                    nonloc_lastframereached = True
            psim.SameLine()  # type: ignore
            if psim.Button("Skip to last show() frame"):  # type: ignore
                tuple(stepper)
                nonloc_lastframereached = True

            if nonloc_lastframereached:
                psim.TextUnformatted("(that was the last show frame)")  # type: ignore

            if recording_npz_comment:
                if psim.TreeNode("Comment in recording"):  # type: ignore
                    psim.TextWrapped(recording_npz_comment)  # type: ignore
                    psim.TreePop()  # type: ignore

            psim.End()  # type: ignore

        if draw_playback_ui:
            next(stepper)
            ps.set_open_imgui_window_for_user_callback(False)
            ps.set_user_callback(__ps_callback)
            ps.show()
        else:
            tuple(stepper)


def replay_viewer_main():
    import argparse

    thlog = Thlogger(LOG_INFO, VIZ_INFO, "polyscope replay")

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["replay", "comment", "get_arrays", "get_images"])
    parser.add_argument("recording_filename", type=str)
    parser.add_argument("-s", "--screenshot_filename_template", type=str)
    array_id_arg = parser.add_argument(
        "-ai",
        "--array_ids",
        type=int,
        nargs="+",
        help="only valid for, and is required by, 'get_arrays' and 'get_images' mode. saves out the saved numpy arrays in the recording with these ids -ai/--array_id to -as/--array_save",
    )
    array_save_arg = parser.add_argument(
        "-as",
        "--array_save",
        type=str,
        help="only valid for, and is required by, 'get_arrays' mode. path to save the extracted array to",
    )
    parser.add_argument(
        "-ac",
        "--array_comment",
        type=str,
        help="only valid for 'get_arrays' mode. comment to save to the npz file that the extracted arrays are saved to.",
    )
    image_save_filename_template_arg = parser.add_argument(
        "-is",
        "--image_save_filename_template",
        type=str,
        help="only valid for 'get_images' mode. filename template for every saved image, where the image array ID (and label if present) is represented by {}",
    )
    namespace = parser.parse_args()
    mode = namespace.mode
    fname = namespace.recording_filename
    if mode == "replay":
        thlog.play_ps_recording(fname, namespace.screenshot_filename_template, True)
    elif mode == "comment":
        # print the comment in the recording
        with np.load(fname) as npz:
            if comment := npz.get("comment", None):
                print(str(comment))
    elif mode == "get_arrays":
        # save some arrays from the recording to a separate npz file
        if namespace.array_ids is None:
            raise argparse.ArgumentError(
                array_id_arg,
                "for mode 'get_arrays', must specify -ai/--array_ids and -as/--array_save",
            )
        if namespace.array_save is None:
            raise argparse.ArgumentError(
                array_save_arg,
                "for mode 'get_arrays', must specify -ai/--array_ids and -as/--array_save",
            )
        with np.load(fname) as npz:
            extracted_arrays = {}
            for array_id in namespace.array_ids:
                extracted_array = npz.get(_make_tagged_array_key(array_id), None)
                if extracted_array is None:
                    raise IndexError(
                        f"array id {array_id} not present in the recording's saved arrays!"
                    )
                extracted_arrays[str(array_id)] = extracted_array
            # comment in the npz file of extracted arrays
            if namespace.array_comment:
                extracted_arrays["comment"] = np.array(str(namespace.array_comment))

        np.savez_compressed(namespace.array_save, **extracted_arrays)
    elif mode == "get_images":
        from PIL import Image

        # save some images from the recording to files. requires PIL
        if namespace.array_ids is None:
            raise argparse.ArgumentError(
                array_id_arg,
                "for mode 'get_images', must specify -ai/--array_ids and -is/--image_save_filename_template",
            )
        if namespace.image_save_filename_template is None:
            raise argparse.ArgumentError(
                image_save_filename_template_arg,
                "for mode 'get_images', must specify -ai/--array_ids and -is/--image_save_filename_template",
            )
        with np.load(fname) as npz:
            metadata_dicts__arr = npz.get("metadata_dicts")
            metadata_dicts: Optional[Dict[str, dict]] = (
                json.loads(str(metadata_dicts__arr))
                if metadata_dicts__arr is not None
                else None
            )
            extracted_arrays = {}
            for array_id in namespace.array_ids:
                array_key = _make_tagged_array_key(array_id)
                extracted_array: Optional[np.ndarray] = npz.get(array_key, None)
                if extracted_array is None:
                    raise IndexError(
                        f"array id {array_id} not present in the recording's saved arrays!"
                    )
                extracted_array_meta = (
                    metadata_dicts.get(array_key) if metadata_dicts else None
                )
                image_save_fname = namespace.image_save_filename_template.format(array_id)
                __save_fn = lambda arr: Image.fromarray(arr).save(image_save_fname)
                if extracted_array_meta:
                    if pil_mode := extracted_array_meta.get("pngbytes"):
                        # the array is a byte array that can be written as a png file
                        thlog.info(
                            f"array {array_id} is stored as png bytes, saving directly"
                        )
                        __save_fn = lambda arr: arr.tofile(image_save_fname)
                __save_fn(extracted_array)
    else:
        raise ValueError(f"unknown script mode {mode}")


if __name__ == "__main__":
    replay_viewer_main()
