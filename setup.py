import os
import subprocess
from setuptools import setup

MAJOR               = 0
MINOR               = 1
PATCH               = 0
ISRELEASED          = False
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, PATCH)

build_type = 'prod'
#build_type = 'prof'
#build_type = 'debug'

def git_version():
    """Return the git revision as a string (adapted from NUMPY source)

    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C++'
        env['LANG'] = 'C++'
        env['LC_ALL'] = 'C++'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def get_version_info():
    """ Return version info (adapted from NUMPY source) """
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pyfvcom2/version.py'):
        # must be a source distribution, use existing version file
        try:
            from pyfvcom2.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing " \
                              "pyfvcom2/version.py and the build directory " \
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION

def write_version_py(filename='pyfvcom2/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM PyFVCOM2 SETUP.PY
#
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    with open(filename, 'w') as a:
        try:
            a.write(cnt % {'version': VERSION,
                           'full_version' : FULLVERSION,
                           'git_revision' : GIT_REVISION,
                           'isrelease': str(ISRELEASED)})
        finally:
            pass

# Rewrite the version file everytime
write_version_py()

setup(version=get_version_info()[0]
)
