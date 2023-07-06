import ctypes
from pathlib import Path

from threadpoolctl import LibController, register


path = Path(__file__).parent / "my_threaded_lib.so"
ctypes.CDLL(path)


class MyThreadedLibController(LibController):
    user_api = "my_threaded_lib"
    internal_api = "my_threaded_lib"
    filename_prefixes = ("my_threaded_lib",)

    def get_num_threads(self):
        return getattr(self._dynlib, "mylib_get_num_threads")()

    def set_num_threads(self, num_threads):
        getattr(self._dynlib, "mylib_set_num_threads")(num_threads)

    def get_version(self):
        get_version = getattr(self._dynlib, "mylib_get_version")
        get_version.restype = ctypes.c_char_p
        return get_version().decode("utf-8")

    def set_additional_attributes(self):
        self.some_attr = "some_value"


register(MyThreadedLibController)
