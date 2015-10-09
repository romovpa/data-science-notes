#!/bin/sh

PYTHONPATH=/Users/romovpa/workspace/bigartm/src/python/examples/src/python:$PYTHONPATH
ARTM_SHARED_LIBRARY=/Users/romovpa/workspace/bigartm/src/python/examples/build/src/artm/libartm.dylib

ipython notebook --port 4000 --no-browser --ip \*
