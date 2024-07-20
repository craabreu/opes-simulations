#!/bin/bash
python run.py --method opes --varfreq 50 --incomingbw
python run.py --method opes --varfreq 50 --uncompressed
python run.py --method opes-explore --varfreq 50 --unreweighted
python run.py --method opes-explore --varfreq 01
python run.py --method opes --varfreq 01
python run.py --method opes-explore --varfreq 50
python run.py --method opes-explore --varfreq 50 --uncompressed
python run.py --method opes-explore --varfreq 25
python run.py --method opes-explore --varfreq 50 --unreweighted --uncorrected
python run.py --method opes-explore --varfreq 50 --bounded
python run.py --method opes-explore --varfreq 50 --incomingbw
python run.py --method metad
python run.py --method opes --varfreq 25
python run.py --method opes --varfreq 50
python run.py --method opes --varfreq 50 --bounded
