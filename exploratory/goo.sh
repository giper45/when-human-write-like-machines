#!/bin/bash

../venv/bin/python run_generate.py model=gemma2_9b
telegram-notify "completed unina gemma xsum"
../venv/bin/python run_generate.py dataset=wp model=gemma2_9b
telegram-notify "completed unina gemma wp"
../venv/bin/python run_generate.py dataset=owt model=gemma2_9b
telegram-notify "completed unina gemma owt"
