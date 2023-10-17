import ctypes
from pathlib import Path

from threadpoolctl import LibController, register


path = Path(__file__).parent / "my_threaded_lib.so"
ctypes.CDLL(path)


class MyThreadedLibController(LibController):
    # names for threadpoolctl's context filtering
    user_api = "my_threaded_lib"
    internal_api = "my_threaded_lib"

    # Patterns to identify the name of the linked library to load.
    # If a dynamic library with a matching filename is linked to the python
    # process, it will be loaded as the `dynlib` attribute of the LibController
    # instance.
    filename_prefixes = ("my_threaded_lib",)

    # (Optional) Symbols that the linked library is expected to expose. It is used along
    # with the `filename_prefixes` to make sure that the correct library is identified.
    check_symbols = (
        "mylib_get_num_threads",
        "mylib_set_num_threads",
        "mylib_get_version",
    )

    def get_num_threads(self):
        # This function should return the current maximum number of threads,
        # which is reported as "num_threads" by `ThreadpoolController.info`.
        return getattr(self.dynlib, "mylib_get_num_threads")()

    def set_num_threads(self, num_threads):
        # This function limits the maximum number of threads,
        # when `ThreadpoolController.limit` is called.
        getattr(self.dynlib, "mylib_set_num_threads")(num_threads)

    def get_version(self):
        # This function returns the version of the linked library if it is exposed,
        # which is reported as "version" by `ThreadpoolController.info`.
        get_version = getattr(self.dynlib, "mylib_get_version")
        get_version.restype = ctypes.c_char_p
        return get_version().decode("utf-8")

    def set_additional_attributes(self):
        # This function is called during the initialization of the LibController.
        # Additional information meant to be exposed by `ThreadpoolController.info`
        # should be set here as attributes of the LibController instance.
        self.some_attr = "some_value"


register(MyThreadedLibController)
