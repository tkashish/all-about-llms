# Attention — the motivation

## What is it in simple English?

Attention is how each word in a sentence **looks at the other words** and
pulls in information from them.

## What problem does it solve?

After the embedding step, each token has its own vector from the lookup
table. But that vector is **context-free** — the table has one row per
token, so the same word always starts as the same vector.

Example:
- "river **bank**" — bank means shore
- "**bank** account" — bank means financial institution

Same token "bank", same embedding row, completely different meaning.
The model must be able to figure out which meaning applies from the
surrounding words.

Attention lets the vector for "bank" mix in information from its
neighbors, so the vector ends up different depending on context.

## Where attention sits in the pipeline

```
text
  ↓ tokenize
token IDs
  ↓ embed
vectors (one per token, context-free)
  ↓ ATTENTION  ← fixes the context-free problem
vectors (one per token, context-aware)
  ↓ MLP, more attention layers, ...
  ↓ output head
probability over next token
```

Attention runs on every forward pass of the model.

## High-level mechanism (covered in next notes)

For each token:
1. Compute a **relevance weight** between it and every other token.
2. Build a new vector as a **weighted average** of all token vectors,
   using those weights.

That new vector is contextualized — it carries information from all the
tokens that were relevant to it.

The specifics of how those relevance weights are computed = **Q, K, V**
and the scaled dot-product formula, coming next.
