""" PyFVCOM2 exceptions

The intention here is to help callers distinguish between expected
PyFVCOM2 exceptions and regular python exceptions, which indicate a
bug in PyFVCOM2's API.
"""


class PyFVCOM2Exception(Exception):
    pass


class PyFVCOM2RuntimeError(PyFVCOM2Exception):
    pass


class PyFVCOM2AttributeError(PyFVCOM2Exception):
    pass


class PyFVCOM2ValueError(PyFVCOM2Exception):
    pass


class PyFVCOM2TypeError(PyFVCOM2Exception):
    pass
