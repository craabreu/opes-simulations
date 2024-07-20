#!/bin/bash
python run.py --method opes --incomingbw
python run.py --method opes --uncompressed
python run.py --method opes-explore --unreweighted
python run.py --method opes-explore
python run.py --method opes-explore --uncompressed
python run.py --method opes-explore --unreweighted --uncorrected
python run.py --method opes-explore --bounded
python run.py --method opes-explore --bounded --unreweighted
python run.py --method opes-explore --bounded --unreweighted --uncorrected
python run.py --method opes-explore --incomingbw
python run.py --method metad
python run.py --method opes
python run.py --method opes --bounded
