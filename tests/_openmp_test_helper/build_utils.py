import os
import sys


def set_cc_variables(var_name="CC"):
    cc_var = os.environ.get(var_name)
    if cc_var is not None:
        os.environ["CC"] = cc_var
        if sys.platform == "darwin":
            os.environ["LDSHARED"] = (
                cc_var + " -bundle -undefined dynamic_lookup")
        else:
            os.environ["LDSHARED"] = cc_var + " -shared"

    return cc_var
