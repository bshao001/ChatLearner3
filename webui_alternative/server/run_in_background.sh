#!/bin/bash
export PYTHONPATH="/work/chatbot/ChatLearner/"
nohup python chatservice.py > sal.out 2> err.out &

