#!/usr/bin/env python3 -u

from contextlib import contextmanager
import subprocess
import sys
import argparse
import logging
import os
import fcntl
import requests

logging.basicConfig(format='%(levelname)s %(msg)s',
                    level=logging.INFO)

ap = argparse.ArgumentParser(
    description='Run a MyCal experiment with iterative training and testing (webcal version)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

ap.add_argument('--host',
                default='127.0.0.1',
                help='Host for webcal')
ap.add_argument('--port',
                default='8123',
                help='Port for webcal')
ap.add_argument('-t', '--train_rank',
                default=10,
                type=int,
                help='Maximum pool rank for initial training')
ap.add_argument('-n', '--num_docs',
                default=1,
                type=int,
                help='Number of documents to score at each round')
ap.add_argument('-s', '--max_steps',
                default=None,
                type=int,
                help='Number of review steps to take, None if until all found')
ap.add_argument('-f', '--fail-out',
                action='store_true',
                help='Should we stop reviewing at deep-2023 criteria?')
ap.add_argument('-z', '--zero-steps',
                default=None,
                type=int,
                help='Number of steps without a relevant document to do before stopping')
ap.add_argument('-r', '--relstop',
                help='Stop when all relevant documents are found',
                action='store_false')
ap.add_argument('topic',
                help='Topic number, used as prefix for all files')
ap.add_argument('pool',
                help='Pool to take training data from')
ap.add_argument('qrels',
                help='Relevance judgments for training and test')

args = ap.parse_args()
TRAIN_URL = f'http://{args.host}:{args.port}/train'
SCORE_URL = f'http://{args.host}:{args.port}/score'

class Lock:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        
    def acquire(self):
        if args.disable_locking:
            return
        self.file = open(self.filename, 'w')
        self.file = open(self.filename, 'w')
        print(os.getpid(), file=self.file)
        fcntl.flock(self.file, fcntl.LOCK_EX)
    
    def release(self):
        if args.disable_locking:
            return
        if self.file:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None

    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self):
        self.release()

@contextmanager
def locked(filename):
    lock = Lock(filename)
    try:
        lock.acquire()
        yield lock
    finally:
        lock.release()

qrels = {}
with open(args.qrels, 'r') as qrels_file:
    for line in qrels_file:
        topic, _, docid, rel = line.split()
        if topic != args.topic:
            continue
        qrels[docid] = int(rel)
num_rel = sum(1 for v in qrels.values() if v > 0)
rel_seen = 0
docs_reviewed = 0
rel_seen_after_training = 0

def do_initial_training():
    global rel_seen, docs_reviewed
    training_qrels = f'{args.topic}.train.0'
    output_model = f'{args.topic}.model.0'
    with open(training_qrels, 'w') as train_file:
        with open(args.pool, 'r') as pool_file:
            for line in pool_file:
                topic, docid, rank, _ = line.split(maxsplit=3)
                if topic != args.topic:
                    continue
                if int(rank) > args.train_rank:
                    continue
                docs_reviewed += 1
                if docid in qrels:
                    rel = qrels[docid]
                    if rel > 0:
                        rel_seen += 1
                else:
                    rel = 0
                print(f'{topic} 0 {docid} {rel}', file=train_file)

    resp = requests.get(TRAIN_URL, params={'model_file': f'{args.topic}.model.0', 'qrels_file': f'{args.topic}.train.0'})
    if resp.status_code != 200:
        logging.critical(f'Failed to train model: {resp.text}')
        sys.exit(-1)

    print(f'Initial: {docs_reviewed} reviewed, {rel_seen} relevant out of {num_rel} total')

zero_steps = 0
sum_prec = 0.0

def one_step(step):
    global rel_seen, docs_reviewed, zero_steps, rel_seen_after_training
    last_step = step - 1
    if last_step < 0:
        logging.critical(f'Bad step value {step}')
        sys.exit(-1)

    # get last training data
    train = {}
    with open(f'{args.topic}.train.{last_step}', 'r') as train_file:
        for line in train_file:
            topic, _, docid, rel = line.split()
            if topic != args.topic:
                logging.critical('Bad training file, wrong topic')
                sys.exit(-1)
            train[docid] = rel
        
    # Score collection
    resp = requests.get(SCORE_URL, params={'model_file': f'{args.topic}.model.{last_step}',
                                           'num_results': args.num_docs,
                                           'exclude_file': f'{args.topic}.train.{last_step}'
                                           })
    if resp.status_code != 200:
        logging.critical(f'Failed to score collection: {resp.text}')
        sys.exit(-1)

    # Add judgments to training data
    rel_seen_this_step = 0
    for entry in resp.json():
        docid = entry['docid']
        score = entry['score']
        if docid in train:
            logging.critical(f'Scored document {docid} already in training data')
            sys.exit(-1)
        docs_reviewed += 1
        if docid in qrels:
            rel = qrels[docid]
            if rel > 0:
                rel_seen += 1
                rel_seen_this_step += 1
                rel_seen_after_training += 1
                zero_steps = 0
        else:
            rel = 0
        logging.debug(f'Adding {docid} to training')
        train[docid] = rel
        
    if rel_seen_this_step == 0:
        zero_steps += 1
        
    # Write new training data
    with open(f'{args.topic}.train.{step}', 'w') as train_file:
        for docid, rel in train.items():
            print(args.topic, 0, docid, rel, file=train_file)

    # Train new model
    resp = requests.get(TRAIN_URL, params={'model_file': f'{args.topic}.model.{step}', 'qrels_file': f'{args.topic}.train.{step}'})
    if resp.status_code != 200:
        logging.critical(f'Failed to train model at step {step}: {resp.text}')
        sys.exit(-1)

    print(f'{args.topic} Step {step}: {docs_reviewed} reviewed, {rel_seen} relevant / {num_rel} total, {rel_seen_after_training / step:.4f} step set prec')
    return rel_seen_this_step

if __name__ == '__main__':
    step = 0
    do_initial_training()
    sum_prec += rel_seen / docs_reviewed
    print(f'AP: {sum_prec / num_rel:.4f}')
    
    if not args.max_steps:
        args.max_steps = len(qrels)
    
    while True:
        step += 1
        new_rel = one_step(step)
        if new_rel > 0:
            sum_prec += rel_seen / docs_reviewed
        print(f'AP: {sum_prec / num_rel:.4f}')
        
        if args.relstop and rel_seen >= num_rel:
            logging.info('Found all relevant, stopping')
            break
        if args.max_steps and step >= args.max_steps:
            logging.info('Maximum steps, stopping')
            break
        if args.fail_out and docs_reviewed > 300 and float(rel_seen)/docs_reviewed > 0.5:
            logging.info(f'Stop criterion met: seen > 300, rel > 0.5; drop topic')
            break
        if args.fail_out and docs_reviewed > 150 and new_rel < 3 and float(rel_seen)/docs_reviewed < 0.4:
            # can't happen when we are running one doc at a time.
            logging.info(f'Stop criterion met: seen > 150, rel < 0.4; keep topic')
            break
        if args.zero_steps and zero_steps > args.zero_steps:
            logging.info(f'{zero_steps} steps with no new relevant found, stopping')
            break
        
        
