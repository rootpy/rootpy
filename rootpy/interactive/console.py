# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Spawn an interactive python console in the current frame.

For example::

    from rootpy.interactive import interact
    x = 1
    interact()
    # Now you're in a python console.

"""
from __future__ import absolute_import

import code
import readline
import sys

import ROOT as R

from ..logger.magic import fix_ipython_startup

__all__ = [
    'interact',
]

# overridden if importing ipython is successful
have_ipython = False

# Make it so that a subsequent \n has no effect
UP_LINE = '\r\x1b[1A'


def interact_plain(header=UP_LINE, local_ns=None,
                   module=None, dummy=None,
                   stack_depth=1, global_ns=None):
    """
    Create an interactive python console
    """
    frame = sys._getframe(stack_depth)

    variables = {}

    if local_ns is not None:
        variables.update(local_ns)
    else:
        variables.update(frame.f_locals)

    if global_ns is not None:
        variables.update(local_ns)
    else:
        variables.update(frame.f_globals)

    shell = code.InteractiveConsole(variables)
    return shell.interact(banner=header)

try:
    from IPython.terminal.embed import InteractiveShellEmbed
    have_ipython = True

except ImportError:
    interact = interact_plain

else:
    # ROOT has a bug causing it to print (Bool_t)1 to the console.
    # This is fixed in defaults.py if rootpy is imported under the ipython
    # interpreter, but at this point that is too late, so we need to try again
    _finalSetup = getattr(R.__class__, "_ModuleFacade__finalSetup", None)
    if _finalSetup:
        _orig_func = getattr(_finalSetup, "_orig_func", None)
        if _orig_func:
            _finalSetup = _orig_func
        fix_ipython_startup(_finalSetup)

    interact_ipython_ = None

    def interact_ipython(header='', *args, **kwargs):
        global interact_ipython_

        def pre_prompt_hook(_):
            R.gInterpreter.EndOfLineAction()

        # Interact is a callable which starts an ipython shell
        if not interact_ipython_:
            interact_ipython_ = InteractiveShellEmbed(banner1=UP_LINE)
        # needed for graphics to work correctly
        interact_ipython_.set_hook('pre_prompt_hook', pre_prompt_hook)
        stack_depth = kwargs.pop("stack_depth", 0) + 2
        kwargs["stack_depth"] = stack_depth
        interact_ipython_(header, *args, **kwargs)

    interact = interact_ipython
