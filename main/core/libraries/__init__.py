from .stdlib import Stdlibrary
from .loguru import LoguruAdapter
from .structlog import StructlogAdapter
from .logbook import LogbookAdapter
from .pythonjsonlogger import PythonJsonLoggerAdapter
from .picologging import PicoLoggingAdapter
from .aiologger import Aiologger
# Only synchronous libraries here
LIBRARIES = {
    "stdlib": Stdlibrary,
    "loguru": LoguruAdapter,
    "structlog": StructlogAdapter,
    "logbook": LogbookAdapter,
    "pythonjsonlogger": PythonJsonLoggerAdapter,
    "picologging": PicoLoggingAdapter,
    #"aiologger":Aiologger
}


#try:
    #from .eliot import run_one as eliot_run_one
 #   LIBRARIES["eliot"] = eliot_run_one
#except Exception as e:
#    import traceback
 #   print("[WARN] Skipping eliot adapter import:")
  #  print(traceback.format_exc())

    
