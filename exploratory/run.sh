#!/bin/bash

../venv/bin/python run_freellm.py dataset=xsum 
telegram-notify "secsi-gpu - completed xsum run_freellm.py dataset=xsum"
../venv/bin/python run_freellm.py dataset=xsum model=mistral7b
telegram-notify "secsi-gpu - completed xsum run_freellm.py dataset=xsum model=mistral7b"
../venv/bin/python run_freellm.py dataset=xsum model=qwen25_7b
telegram-notify "secsi-gpu - completed xsum run_freellm.py dataset=xsum model=qwen25_7b"



../venv/bin/python run_freellm.py dataset=owt
telegram-notify "secsi-gpu - completed owt run_freellm.py dataset=owt"
../venv/bin/python run_freellm.py dataset=owt model=mistral7b
telegram-notify "secsi-gpu - completed owt run_freellm.py dataset=owt model=mistral7b"
../venv/bin/python run_freellm.py dataset=owt model=qwen25_7b
telegram-notify "secsi-gpu - completed owt run_freellm.py dataset=owt model=qwen25_7b"

../venv/bin/python run_freellm.py dataset=wp
telegram-notify "secsi-gpu - completed wp run_freellm.py dataset=wp"
../venv/bin/python run_freellm.py dataset=wp model=mistral7b
telegram-notify "secsi-gpu - completed wp run_freellm.py dataset=wp model=mistral7b"
../venv/bin/python run_freellm.py dataset=wp model=qwen25_7b
telegram-notify "secsi-gpu - completed wp run_freellm.py dataset=wp model=qwen25_7b"