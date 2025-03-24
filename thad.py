# monadic & functional & misc helpers
from typing import Optional, Callable, TypeVar, List, Tuple
from functools import partial, lru_cache
import os
import sys
from contextlib import contextmanager


_T = TypeVar("_T")
_S = TypeVar("_S")


def expect(mx: Optional[_T], exc: Exception) -> _T:
    if mx is None:
        raise exc
    else:
        return mx


def maybe(mx: Optional[_T], f: Callable[[_T], _S], default: _S) -> _S:
    return default if mx is None else f(mx)


def id(x: _T) -> _T:
    return x


def const(x: _T) -> Callable[..., _T]:
    return lambda *args, **kwargs: x


@lru_cache()
def parse_lr_schedule_string_into_lr_lambda(lr_spec: str) -> Callable[[int], float]:
    """
    returns a function (epoch: int) -> lr: float

    format for lr_spec:
    $STARTLR[>$ENDLR][:$NUM_EPOCHS;]*

    for instance,
    0.0003:300; 0.0003>0.0001:100
    will keep the lr at constant 0.0003 for 300 epochs, then ramp down
    linearly from 0.0003 to 0.0001 over the next 100 epochs.

    0.0003
    will just keep the lr at constant 0.0003 for all epochs

    0.0003>0.0002:100;0.0002
    will ramp from 0.0003 to 0.0002 over the first 100 epochs then
    keep constant at 0.0002 for the remaining epochs

    0.0003>0.0002:100
    will ramp from 0.0003 to 0.0002 for 100 epochs then stay at 0.0002

    0.003:3; 0.04>0.02:5; 0.1:2
    will stay at 0.003 for 3 epochs, ramp from 0.04 to 0.02 for 5 epochs,
    then stay at 0.1 for the remainder (despite being specified to stay for just
    2 epochs, the last LR is used as the hold value for all remaining epochs)
    """
    phases = lr_spec.split(";")
    epoch_nums_and_phase_funcs = []
    make_const_func = lambda lr, _: lr
    make_linear_ramp_func = (
        lambda start_lr, end_lr, n_epochs, from_epoch, current_epoch: (
            (end_lr - start_lr) / (n_epochs - 1)
        )
        * (current_epoch - 1 - from_epoch)
        + start_lr
    )

    current_epoch_n = 0
    n_phases = len(phases)
    for phase_i, phase in enumerate(phases):
        phase_split = phase.strip().split(":")
        if len(phase_split) < 2:
            # no n_epochs specified, set to a negative number, later we can let
            # it match all epochs (effectively an infinitely long final phase)
            n_epochs = -1
            last_epoch_this_phase = -1
        else:
            n_epochs = int(phase_split[1])
            last_epoch_this_phase = current_epoch_n + n_epochs

        start_end_lr = phase_split[0]
        start_end_lr_split = start_end_lr.split(">")
        start_lr = float(start_end_lr_split[0])
        end_lr = start_lr
        if len(start_end_lr_split) < 2:
            # constant function
            epoch_nums_and_phase_funcs.append(
                (last_epoch_this_phase, partial(make_const_func, start_lr))
            )
        else:
            # linear ramp function
            if last_epoch_this_phase < 0:
                # ramp phases cannot be infinitely-long phases
                # this branch also means n_epochs == -1 (an invalid value to
                # satisfy the typechecker)
                raise ValueError(
                    f"Error in learning rate schedule spec '{phase}': linear ramp expression must have number of epochs specified"
                )
            end_lr = float(start_end_lr_split[1])
            if n_epochs == 1:
                epoch_nums_and_phase_funcs.append(
                    (last_epoch_this_phase, partial(make_const_func, end_lr))
                )
            else:
                epoch_nums_and_phase_funcs.append(
                    (
                        last_epoch_this_phase,
                        partial(
                            make_linear_ramp_func,
                            start_lr,
                            end_lr,
                            n_epochs,
                            current_epoch_n,
                        ),
                    )
                )

        if last_epoch_this_phase < 0:
            break

        if phase_i == (n_phases - 1):
            # last phase, manually insert an infinite phase holding end_lr
            epoch_nums_and_phase_funcs.append((-1, partial(make_const_func, end_lr)))

        current_epoch_n += n_epochs

    def lr_lambda(
        epoch_nums_and_phase_funcs: List[Tuple[int, Callable[[int], float]]], epoch: int
    ) -> float:
        for last_epoch_this_phase, lambda_func_this_phase in epoch_nums_and_phase_funcs:
            if (last_epoch_this_phase < 0) or (epoch <= last_epoch_this_phase):
                # last_epoch_this_phase < 0 indicates last phase. use this phase
                # if the phase is an infinite last phase or epoch is
                # <=last_epoch_this_phase
                return lambda_func_this_phase(epoch)
        # impossible return, for satisfying typechecker since epoch < 0 is
        # caught and epoch > max epoch defined in that list is understood to be
        # the last valid epoch's lr
        return 0.0

    return partial(lr_lambda, epoch_nums_and_phase_funcs)


def next_increment_path(path_pattern: str):
    """
    from https://stackoverflow.com/a/47087513

    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-{}.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern.format(i)):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern.format(c)) else (a, c)

    return path_pattern.format(b)


# output silencer for prints from C++/C libraries
# from https://stackoverflow.com/a/17954769
@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
