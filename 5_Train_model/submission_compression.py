"""
Author -- Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 07.06.2022

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may  be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form, requires explicit prior
acceptance of the authors.

###############################################################################

This script will compress the specified submission file (SUBMISSION_FILE) with
the given compression algorithm (COMPRESSION, default: "bzip2"). The following
compression algorithms are supported (choose any as COMPRESSION):
> zip (https://docs.python.org/3/library/zipfile.html)
> gzip (https://docs.python.org/3/library/gzip.html)
> bzip2 (https://docs.python.org/3/library/bz2.html)
> lzma (https://docs.python.org/3/library/lzma.html)
The compressed file will be stored at the same location as the specified
submission file, with an appropriate additional compression file extension.

Usage: python submission_compression.py SUBMISSION_FILE [--compression COMPRESSION]
"""

import argparse
import importlib
import shutil


def try_compression(file: str, name: str, module_name: str, extension: str, function: callable):
    try:
        compression_module = importlib.import_module(module_name)
        
        function(file, compression_module, extension)
    except ImportError as ex:
        raise ImportError(f"compression='{name}' failed: required module could not be loaded ({ex})")


def compression_open_context_manager(file: str, module, extension: str):
    with open(file, "rb") as f_in:
        with module.open(file + extension, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def zip_compression(file: str, module, extension: str):
    with module.ZipFile(file + extension, "w", compression=module.ZIP_DEFLATED) as z:
        z.write(file)


modules = {"zip": "zipfile", "gzip": "gzip", "bzip2": "bz2", "lzma": "lzma"}
extensions = {"zip": ".zip", "gzip": ".gz", "bzip2": ".bz2", "lzma": ".xz"}
assert set(modules.keys()) == set(extensions.keys())

parser = argparse.ArgumentParser()
parser.add_argument("submission_file", type=str, help="Path to submission file")
parser.add_argument("--compression", type=str, default="bzip2",
                    help=f"Compression algorithm to use; must be one of {list(modules.keys())}. Default: 'bzip2'")
args = parser.parse_args()
submission_file = args.submission_file
compression = args.compression

if compression not in modules:
    raise ValueError(f"Unknown compression algorithm '{compression}'; must be one of {list(modules.keys())}")

try_compression(
    file=submission_file,
    name=compression,
    module_name=modules[compression],
    extension=extensions[compression],
    function=zip_compression if compression == "zip" else compression_open_context_manager
)
print(f"Successfully compressed '{submission_file}' to '{submission_file + extensions[compression]}' "
      f"using {compression} as compression algorithm")
