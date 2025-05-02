#!/usr/bin/env python3 -u

from contextlib import contextmanager
import subprocess
import sys
import argparse
import logging
import os
import fcntl

logging.basicConfig(format='%(levelname)s %(msg)s',
                    level=logging.DEBUG)

ap = argparse.ArgumentParser(
    description='Run a MyCal experiment with iterative training and testing',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
ap.add_argument('-l', '--lockfile',
                default='lockfile',
                help='Lockfile to use when training')
ap.add_argument('-L', '--disable-locking',
                action='store_true',
                help='Disable collection data lock')
ap.add_argument('-d', '--docdb',
                help='Document database prefix',
                default='cd45')
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
MYCAL = '/Users/soboroff/mycal-project/mycal/target/release/mycal'
SCORE = '/Users/soboroff/mycal-project/mycal/target/release/score-index'

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
    with open(f'{args.topic}.train.0', 'w') as train_file:
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

    with locked(args.lockfile):
        subprocess.run([ MYCAL, args.docdb, f'{args.topic}.model.0', 'train', 
                        f'{args.topic}.train.0' ], check=True),
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
    result = subprocess.run([ SCORE, args.docdb, f'{args.topic}.model.{last_step}', 
                             '-n', str(args.num_docs), 
                             '-e', f'{args.topic}.train.{last_step}'], check=True, capture_output=True)

    # Add judgments to training data
    rel_seen_this_step = 0
    for line in result.stdout.splitlines():
        docid, score = line.decode('utf-8').split()
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
    with locked(args.lockfile):
        subprocess.run( [MYCAL, args.docdb, f'{args.topic}.model.{step}', 'train', 
                    f'{args.topic}.train.{step}' ], check=True)

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
        if args.fail_out and docs_reviewed > 150 and new_rel > 3 and float(rel_seen)/docs_reviewed > 0.4:
            # can't happen when we are running one doc at a time.
            logging.info(f'Stop criterion met: seen > 150, rel < 0.4; keep topic')
            break
        if args.zero_steps and zero_steps > args.zero_steps:
            logging.info(f'{zero_steps} steps with no new relevant found, stopping')
            break
        
        
