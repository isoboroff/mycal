# Features and plans for MyCal

## back-end performance

The major issue here is that when we compute predictions, we currently stream over the entire collection of feature vectors.  The infrastructure for that uses Serde serialization for the on-disk format.  A profile of the prediction phase (using samply) shows that we spend the most time serializing and deserializing.

So two forks:

1. Move to Arrow2 as an on-disk format.  There is much more lightweight transforms from on-disk into memory (basically none, because your objects have to have primitive types).
1. Switch from DAAT to an inverted index search.  The inverted index can also be Arrow based, and the goal is fast random access.  We'll keep the term vectors simple for now, but we can look into skip-coding, quantization, and impact scores later.  Since in an inverted search we will be examining many fewer vectors, we might even be able to stay with serde.

## models

Under the hood MyCal is a logistic regression classifier.  Tokens are hashed so the feature space is fixed at around a million terms.  This is simple to understand, compact, and very fast.  Are there more performant models that are still simple to understand, compact, and fast?

It might be good to clock inference so we have some comparison numbers.

## clean up

There are some dalliances with the column store 'kv', which is a little dated at this point.  Those might be a good point for doing the Arrow port, not sure.  I don't seem to have the experiments around but I remember streaming over the kv was slower than streaming over the raw file.  That makes sense because kv still uses serde for serialization, and what I have now is maybe the least expensive implementation that depends on serde.
